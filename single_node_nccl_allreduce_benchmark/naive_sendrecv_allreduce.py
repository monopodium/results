"""
Naive AllReduce using batched isend/irecv.

Two-shot: ReduceScatter + AllGather, each via a single batch_isend_irecv call.
All-to-all pattern — every rank exchanges with every other rank simultaneously.

Single-node path:
  Flat naive RS+AG across the world (existing behavior).

Multi-node path (LOCAL_WORLD_SIZE < WORLD_SIZE, e.g. 2 nodes x 8 GPUs = 16):
  Hierarchical 3-stage to avoid blasting the slow inter-node link with
  every-pair traffic:
    1) Intra-node RS  — each GPU ends up with its 1/local_ws shard reduced
                         within its own node.
    2) Inter-node AR  — same local_rank across nodes runs a naive all-to-all
                         allreduce on the shard (only (num_nodes-1) sends per
                         GPU cross the network, each of size n/local_ws/num_nodes
                         per peer).
    3) Intra-node AG  — broadcast the fully-reduced shards back within node.

  Total cross-node bytes per GPU: 2*(num_nodes-1)/num_nodes * (n/local_ws)
  vs. flat naive: 2*(ws-1)/ws * n with most traffic crossing the network.

Optimizations over baseline flat:
  1. Contiguous RS recv buffer + torch.sum(dim=0) fused accumulation
     (1 kernel vs npeers-1 sequential +=, less HBM traffic)
  2. AG recv directly into input_tensor (eliminates npeers copy kernels)

Usage:
  torchrun --nproc_per_node=8 --nnodes=1 benchmark_both.py --mode naive
  torchrun --nproc_per_node=8 --nnodes=2 ... benchmark_both.py --mode naive
"""

import os
import torch
import torch.distributed as dist


class NaiveAllReduce:
    """Naive all-to-all allreduce with pre-allocated recv buffer.

    Automatically switches between flat (1 node) and hierarchical (>1 node)
    schedules based on the launcher-provided LOCAL_WORLD_SIZE.
    """

    def __init__(self, max_elements, dtype, device):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_ws = int(os.environ.get("LOCAL_WORLD_SIZE", self.world_size))
        self.local_rank = int(os.environ.get("LOCAL_RANK", self.rank % self.local_ws))
        assert self.world_size % self.local_ws == 0, \
            f"WORLD_SIZE ({self.world_size}) must be a multiple of " \
            f"LOCAL_WORLD_SIZE ({self.local_ws})"
        self.num_nodes = self.world_size // self.local_ws
        self.node_id = self.rank // self.local_ws

        # ── Build subgroups for hierarchical path ──
        # NOTE: dist.new_group must be called collectively on ALL ranks in the
        # world, in the same order, even for groups a rank doesn't belong to.
        self._intra_group = None
        self._inter_group = None
        self._intra_info = None   # cached (rank, ws, global_ranks)
        self._inter_info = None
        self._inter_peer = -1     # global rank of sole peer (only when num_nodes==2)
        if self.num_nodes > 1:
            for nid in range(self.num_nodes):
                ranks = list(range(nid * self.local_ws, (nid + 1) * self.local_ws))
                g = dist.new_group(ranks=ranks)
                if nid == self.node_id:
                    self._intra_group = g
            for lr in range(self.local_ws):
                ranks = [nid * self.local_ws + lr for nid in range(self.num_nodes)]
                g = dist.new_group(ranks=ranks)
                if lr == self.local_rank:
                    self._inter_group = g

            self._intra_info = (dist.get_rank(self._intra_group),
                                dist.get_world_size(self._intra_group),
                                dist.get_process_group_ranks(self._intra_group))
            self._inter_info = (dist.get_rank(self._inter_group),
                                dist.get_world_size(self._inter_group),
                                dist.get_process_group_ranks(self._inter_group))
            if self.num_nodes == 2:
                ir_rank, _, ir_globals = self._inter_info
                self._inter_peer = ir_globals[1 - ir_rank]

        # ── Pre-allocate a recv buffer large enough for the worst stage ──
        if self.num_nodes > 1:
            intra_C = max_elements // self.local_ws
            inter_C = intra_C // self.num_nodes
            intra_buf = (self.local_ws - 1) * intra_C
            inter_buf = (self.num_nodes - 1) * inter_C
            buf_size = max(intra_buf, inter_buf)
        else:
            buf_size = (self.world_size - 1) * (max_elements // self.world_size)
        self.recv_buf = torch.empty(max(1, buf_size), dtype=dtype, device=device)

    # ─────────────────────── helpers ───────────────────────

    def _group_info(self, group):
        """Return (group_rank, group_ws, list_of_global_ranks_in_group)."""
        if group is None:
            ws = self.world_size
            return self.rank, ws, list(range(ws))
        if group is self._intra_group:
            return self._intra_info
        if group is self._inter_group:
            return self._inter_info
        return (dist.get_rank(group),
                dist.get_world_size(group),
                dist.get_process_group_ranks(group))

    def _rs(self, tensor, group):
        """Reduce-scatter stage. After this call, tensor[rank*C:(rank+1)*C]
        holds the sum of that chunk across all members of `group`."""
        rank, ws, global_ranks = self._group_info(group)
        if ws <= 1:
            return
        n = tensor.numel()
        C = n // ws
        npeers = ws - 1
        recv = self.recv_buf[:npeers * C]

        ops = []
        buf_i = 0
        for peer in range(ws):
            if peer == rank:
                continue
            peer_global = global_ranks[peer]
            ops.append(dist.P2POp(dist.isend,
                                  tensor[peer * C:(peer + 1) * C],
                                  peer_global, group=group))
            ops.append(dist.P2POp(dist.irecv,
                                  recv[buf_i * C:(buf_i + 1) * C],
                                  peer_global, group=group))
            buf_i += 1
        for r in dist.batch_isend_irecv(ops):
            r.wait()

        my = rank * C
        if npeers == 1:
            tensor[my:my + C].add_(recv[:C])
        else:
            tensor[my:my + C].add_(recv.view(npeers, C).sum(dim=0))

    def _ag(self, tensor, group):
        """All-gather stage. Broadcasts tensor[rank*C:(rank+1)*C] out to
        every other member's corresponding slot."""
        rank, ws, global_ranks = self._group_info(group)
        if ws <= 1:
            return
        n = tensor.numel()
        C = n // ws
        my = rank * C
        my_shard = tensor[my:my + C]

        ops = []
        for peer in range(ws):
            if peer == rank:
                continue
            peer_global = global_ranks[peer]
            ops.append(dist.P2POp(dist.isend, my_shard, peer_global, group=group))
            ops.append(dist.P2POp(dist.irecv,
                                  tensor[peer * C:(peer + 1) * C],
                                  peer_global, group=group))
        for r in dist.batch_isend_irecv(ops):
            r.wait()

    def _flat_ar(self, tensor, group):
        """Full naive allreduce (RS + AG) within `group`."""
        self._rs(tensor, group)
        self._ag(tensor, group)

    def _inter_exchange_2nodes(self, shard):
        """Specialised stage 2 for num_nodes == 2: send/recv + add.

        Saves one batch_isend_irecv call over the generic RS+AG path
        (1 batch call here vs. 2 for RS+AG). Cross-node bytes are identical.

        The shard is split into 2 halves issued in the SAME batch call — this
        keeps per-message size equal to the RS+AG path (half shard) so NCCL's
        network scheduling / channelisation does not shift to a slower regime
        at certain operating points (observed regression at 64 MB when using a
        single full-shard message)."""
        C = shard.numel()
        half = C // 2
        recv = self.recv_buf[:C]
        peer = self._inter_peer
        grp = self._inter_group
        if half == 0:
            # Tiny shard — single message fine.
            ops = [
                dist.P2POp(dist.isend, shard, peer, group=grp),
                dist.P2POp(dist.irecv, recv[:C], peer, group=grp),
            ]
        else:
            ops = [
                dist.P2POp(dist.isend, shard[:half],    peer, group=grp),
                dist.P2POp(dist.irecv, recv[:half],     peer, group=grp),
                dist.P2POp(dist.isend, shard[half:],    peer, group=grp),
                dist.P2POp(dist.irecv, recv[half:C],    peer, group=grp),
            ]
        for r in dist.batch_isend_irecv(ops):
            r.wait()
        shard.add_(recv[:C])

    # ─────────────────────── main entry ───────────────────────

    def __call__(self, input_tensor):
        if self.num_nodes <= 1:
            # Single-node flat path — same pattern as before.
            self._flat_ar(input_tensor, None)
            return input_tensor

        # Multi-node hierarchical path.
        n = input_tensor.numel()
        C_node = n // self.local_ws

        # 1) intra-node RS: reduce the local_rank-th shard within each node
        self._rs(input_tensor, self._intra_group)

        # 2) inter-node AR on that shard (only same-local_rank GPUs talk)
        shard = input_tensor[self.local_rank * C_node:
                             (self.local_rank + 1) * C_node]
        if self.num_nodes == 2:
            self._inter_exchange_2nodes(shard)
        else:
            self._flat_ar(shard, self._inter_group)

        # 3) intra-node AG: broadcast fully-reduced shards back within node
        self._ag(input_tensor, self._intra_group)
        return input_tensor


_naive_instance = None


def naive_sendrecv_allreduce(input_tensor):
    global _naive_instance
    n = input_tensor.numel()
    ws = dist.get_world_size()
    local_ws = int(os.environ.get("LOCAL_WORLD_SIZE", ws))
    num_nodes = ws // local_ws

    # Compute required buffer size for this n with the active schedule.
    if num_nodes > 1:
        intra_C = n // local_ws
        inter_C = intra_C // num_nodes
        required = max((local_ws - 1) * intra_C,
                       (num_nodes - 1) * inter_C)
    else:
        required = (ws - 1) * (n // ws)

    if _naive_instance is None or _naive_instance.recv_buf.numel() < required:
        _naive_instance = NaiveAllReduce(n, input_tensor.dtype, input_tensor.device)
    return _naive_instance(input_tensor)
