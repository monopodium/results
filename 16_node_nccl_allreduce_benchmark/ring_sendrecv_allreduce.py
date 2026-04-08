"""
AllReduce via isend/irecv with three algorithm paths, picked by message size:

  * Halving-doubling (Rabenseifner) — small/medium messages.
      log2(N) reduce-scatter rounds + log2(N) all-gather rounds = only 8
      sequential round-trips for N=16, vs 30 for a ring. Same total bytes,
      ~4x fewer latency hits. Wins when per-round latency dominates.

  * Single-channel ring — fallback when halving-doubling isn't applicable
      (non-power-of-2 world size, or message above the HD scratch cap).

  * Multi-channel parallel ring — large messages.
      Splits the tensor into K slices and runs K independent rings
      concurrently on K NCCL subgroups + CUDA streams. This is the same
      "channels" trick NCCL uses internally to drive the network closer
      to peak bandwidth than a single comm can.

Other optimizations vs. naive ring:
  * Ring Phase 2 receives directly into the destination chunk (no copy).
  * Chunk slice views precomputed once per call.

Tunables: env vars RING_NUM_CHANNELS (default 4) and HD_MAX_MB (default 128).

Usage:
  torchrun --nproc_per_node=8 --nnodes=2 ... benchmark_both.py --mode ring
"""

import os
import torch
import torch.distributed as dist


_DEFAULT_NUM_CHANNELS = int(os.environ.get("RING_NUM_CHANNELS", "4"))
# Tensors at or below this byte size use the halving-doubling path. The HD
# scratch buffer is sized to hold half of the largest HD-eligible tensor.
_HD_MAX_BYTES = int(os.environ.get("HD_MAX_MB", "128")) * 1024 * 1024


class RingAllReduce:
    def __init__(self, max_elements, dtype, device, world_size,
                 num_channels=_DEFAULT_NUM_CHANNELS):
        self.world_size = world_size
        self.num_channels = num_channels
        self.device = device
        self.dtype = dtype

        # Create K subgroups, each containing all ranks. dist.new_group is
        # collective — every rank must call it the same number of times in
        # the same order. Each call yields a fresh NCCL communicator, so
        # ops on different groups can run truly concurrently on different
        # streams. This is the same trick NCCL uses internally for channels.
        all_ranks = list(range(world_size))
        self.groups = [dist.new_group(ranks=all_ranks) for _ in range(num_channels)]
        self.streams = [torch.cuda.Stream(device=device) for _ in range(num_channels)]

        # Per-channel Phase-1 recv buffer. We pick K dynamically per call,
        # and the K=1 fast path uses chunk = max_elements/ws (the largest
        # case). Size every buffer for that worst case so any K works.
        max_chunk = (max_elements + world_size - 1) // world_size
        self.recv_bufs = [
            torch.empty(max_chunk, dtype=dtype, device=device)
            for _ in range(num_channels)
        ]

        # Halving-doubling scratch (one per channel). Each scratch holds the
        # recv half at each HD step on its slice. With K channels handling a
        # tensor of size n, each channel runs HD on n/K elements; step 0 of
        # that per-channel HD sends/recvs n/(2K). The dispatcher always
        # picks the largest K that fits, so the worst-case per-channel
        # half-step is _HD_MAX_BYTES / num_channels / 2. The dispatcher
        # also re-checks the fit and drops K if needed, so this sizing is
        # safe even if a user shrinks num_channels.
        elem_bytes = torch.empty(0, dtype=dtype).element_size()
        hd_scratch_bytes = max(_HD_MAX_BYTES // num_channels // 2, 1)
        hd_scratch_elem = (hd_scratch_bytes + elem_bytes - 1) // elem_bytes
        self.hd_scratchs = [
            torch.empty(hd_scratch_elem, dtype=dtype, device=device)
            for _ in range(num_channels)
        ]
        # Power-of-two world size is required for halving-doubling.
        self._ws_is_pow2 = (world_size & (world_size - 1)) == 0
        self._log2_ws = world_size.bit_length() - 1 if self._ws_is_pow2 else 0

    def _ring_one_channel(self, slice_tensor, group, recv_buf, stream):
        """Run a full ReduceScatter+AllGather ring on one slice on `stream`."""
        rank = dist.get_rank()
        ws = self.world_size
        n = slice_tensor.numel()
        C = n // ws
        left = (rank - 1) % ws
        right = (rank + 1) % ws
        recv_buf = recv_buf[:C]

        chunks = [slice_tensor.narrow(0, i * C, C) for i in range(ws)]

        isend = dist.isend
        irecv = dist.irecv
        P2POp = dist.P2POp
        batch = dist.batch_isend_irecv

        with torch.cuda.stream(stream):
            # Phase 1: ReduceScatter
            si = rank
            ri = (rank - 1) % ws
            for _ in range(ws - 1):
                reqs = batch([
                    P2POp(isend, chunks[si], right, group),
                    P2POp(irecv, recv_buf, left, group),
                ])
                for r in reqs:
                    r.wait()
                chunks[ri].add_(recv_buf)
                si = ri
                ri = (ri - 1) % ws

            # Phase 2: AllGather (recv directly into the destination chunk).
            si = (rank + 1) % ws
            ri = rank
            for _ in range(ws - 1):
                reqs = batch([
                    P2POp(isend, chunks[si], right, group),
                    P2POp(irecv, chunks[ri], left, group),
                ])
                for r in reqs:
                    r.wait()
                si = ri
                ri = (ri - 1) % ws

    def _halving_doubling_one_channel(self, slice_tensor, group, scratch, stream):
        """Rabenseifner halving-doubling allreduce on one slice / one comm.

        Reduce-scatter via recursive halving (log2(N) rounds): at round s
        each rank exchanges with peer = rank XOR (1<<s), sends half of its
        currently-owned range, and reduces the received half into the half
        it keeps. After log2(N) rounds each rank owns 1/N of the slice,
        fully reduced.

        All-gather via recursive doubling (log2(N) rounds, reverse order):
        at round s each rank exchanges its currently-owned range with peer
        = rank XOR (1<<s), doubling the owned range each round.

        Total rounds: 2 * log2(N) (= 8 for N=16) vs 2*(N-1) (= 30) for the
        ring. Same total bytes per rank but far fewer latency hits.
        """
        rank = dist.get_rank()
        n = slice_tensor.numel()
        log2_ws = self._log2_ws

        P2POp = dist.P2POp
        isend = dist.isend
        irecv = dist.irecv
        batch = dist.batch_isend_irecv

        with torch.cuda.stream(stream):
            # Reduce-Scatter (halving). Track currently-owned range as [lo, lo+size).
            lo = 0
            size = n
            for s in range(log2_ws):
                peer = rank ^ (1 << s)
                size //= 2
                if (rank >> s) & 1 == 0:
                    # Keep lower half, send upper half.
                    send_start = lo + size
                    keep_start = lo
                else:
                    # Keep upper half, send lower half.
                    send_start = lo
                    keep_start = lo + size
                send_buf = slice_tensor.narrow(0, send_start, size)
                recv_buf = scratch[:size]
                reqs = batch([
                    P2POp(isend, send_buf, peer, group),
                    P2POp(irecv, recv_buf, peer, group),
                ])
                for r in reqs:
                    r.wait()
                slice_tensor.narrow(0, keep_start, size).add_(recv_buf)
                lo = keep_start

            # All-Gather (doubling) — undo the halving in reverse order.
            for s in range(log2_ws - 1, -1, -1):
                peer = rank ^ (1 << s)
                if (rank >> s) & 1 == 0:
                    recv_start = lo + size
                else:
                    recv_start = lo - size
                send_buf = slice_tensor.narrow(0, lo, size)
                recv_buf = slice_tensor.narrow(0, recv_start, size)
                reqs = batch([
                    P2POp(isend, send_buf, peer, group),
                    P2POp(irecv, recv_buf, peer, group),
                ])
                for r in reqs:
                    r.wait()
                lo = min(lo, recv_start)
                size *= 2

    def _pick_hd_channels(self, nbytes):
        """Pick K for the halving-doubling path.

        Single-channel HD wins on small messages where latency dominates.
        Above a few MB it becomes bandwidth-bound on one comm; running K
        parallel HDs on K subgroups multiplies effective bandwidth, just
        like multi-channel ring. Each per-channel HD still has only 8
        round-trips for ws=16, so we keep the latency win.
        """
        K_max = self.num_channels
        if nbytes <= 8 * 1024 * 1024:        # <=  8 MB: latency-bound, K=1
            return 1
        if nbytes <= 32 * 1024 * 1024:       # <= 32 MB: K=2
            return min(2, K_max)
        return K_max                          # 32 MB <  n <= HD_MAX: K=4

    def _pick_num_channels(self, nbytes):
        """Pick the effective channel count for a given message size.

        Multi-channel parallel rings improve large-message bandwidth, but
        each extra channel pays a per-step latency tax (more ring steps
        per byte, smaller p2p messages, more kernel launches). For small
        messages a single ring wins; for large messages K=4 closes the
        gap to NCCL. Thresholds were chosen empirically on a 2-node x
        8-GPU setup with NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1.
        """
        K_max = self.num_channels
        if nbytes < 32 * 1024 * 1024:        # < 32 MB
            return 1
        if nbytes < 128 * 1024 * 1024:       # < 128 MB
            return min(2, K_max)
        return K_max

    def _dispatch_hd(self, input_tensor, K_hd):
        """Run halving-doubling, optionally split across K_hd parallel channels."""
        n = input_tensor.numel()
        if K_hd == 1:
            # Fast path: no fork/join overhead.
            self._halving_doubling_one_channel(
                input_tensor, self.groups[0], self.hd_scratchs[0],
                torch.cuda.current_stream(),
            )
            return

        slice_size = n // K_hd
        slices = [input_tensor.narrow(0, k * slice_size, slice_size) for k in range(K_hd)]
        current = torch.cuda.current_stream()
        for k in range(K_hd):
            self.streams[k].wait_stream(current)
        for k in range(K_hd):
            self._halving_doubling_one_channel(
                slices[k], self.groups[k], self.hd_scratchs[k], self.streams[k],
            )
        for k in range(K_hd):
            current.wait_stream(self.streams[k])

    def __call__(self, input_tensor):
        n = input_tensor.numel()
        ws = self.world_size
        nbytes = n * input_tensor.element_size()

        # Halving-doubling fast path for small/medium messages. Requires
        # power-of-2 ws and n divisible by (K_hd * ws) so every halving
        # step on every per-channel slice is exact. Multi-channel HD
        # combines HD's low latency (8 rounds for ws=16) with the
        # bandwidth gains of running K independent comms in parallel.
        if self._ws_is_pow2 and nbytes <= _HD_MAX_BYTES:
            K_hd = self._pick_hd_channels(nbytes)
            # Drop K_hd until divisibility holds and per-channel scratch fits.
            while K_hd >= 1:
                if n % (K_hd * ws) == 0:
                    per_channel_n = n // K_hd
                    if (per_channel_n // 2) <= self.hd_scratchs[0].numel():
                        break
                K_hd //= 2
            if K_hd >= 1:
                self._dispatch_hd(input_tensor, K_hd)
                return input_tensor

        K = self._pick_num_channels(nbytes)

        # Need n divisible by K*ws for clean splits; if not, drop K until
        # it is (K=1 always works as long as n % ws == 0).
        while K > 1 and n % (K * ws) != 0:
            K //= 2

        if K == 1:
            # Fast path: no stream forking, no fork/join syncs. This is the
            # cheapest option for small messages where launch overhead and
            # the per-step latency dominate.
            self._ring_one_channel(
                input_tensor, self.groups[0], self.recv_bufs[0],
                torch.cuda.current_stream(),
            )
            return input_tensor

        slice_size = n // K
        slices = [input_tensor.narrow(0, k * slice_size, slice_size) for k in range(K)]

        # Fork: each channel stream waits for the producer (current stream)
        # so it sees the input data, then runs its ring independently.
        current = torch.cuda.current_stream()
        for k in range(K):
            self.streams[k].wait_stream(current)

        for k in range(K):
            self._ring_one_channel(
                slices[k], self.groups[k], self.recv_bufs[k], self.streams[k],
            )

        # Join: current stream waits for every channel to finish before the
        # caller sees the reduced result.
        for k in range(K):
            current.wait_stream(self.streams[k])

        return input_tensor


_ring_instance = None


def ring_sendrecv_allreduce(input_tensor):
    global _ring_instance
    n = input_tensor.numel()
    ws = dist.get_world_size()
    # Worst-case chunk is K=1: full ring chunk = n/ws.
    needed_chunk = (n + ws - 1) // ws
    if _ring_instance is None or _ring_instance.recv_bufs[0].numel() < needed_chunk:
        # NOTE: re-instantiation calls dist.new_group, which is collective.
        # All ranks must hit this path together — true in this benchmark
        # because every rank issues the same sequence of sizes.
        _ring_instance = RingAllReduce(n, input_tensor.dtype, input_tensor.device, ws)
    return _ring_instance(input_tensor)
