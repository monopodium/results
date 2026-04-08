"""
Naive AllReduce using batched isend/irecv.

Two-shot: ReduceScatter + AllGather, each via a single batch_isend_irecv call.
All-to-all pattern — every rank exchanges with every other rank simultaneously.

Single-node path:
  Flat naive RS+AG across the world.

Multi-node small/medium sizes (hierarchical, serial):
  Three stages — intra-RS → inter-AR(shard) → intra-AG.

Multi-node LARGE sizes (hierarchical, pipelined with 2 intra PGs):
  Split input into halves A, B. Run the 3-stage schedule on independent
  intra-node process groups so stage 2 (inter) of one half overlaps with
  stage 1/3 (intra) of the other half. Timeline:

      t0:  RS_A  (pg_intra_a)
      t1:  Ex_A  (pg_inter)   ||  RS_B  (pg_intra_b)
      t2:  AG_A  (pg_intra_a) ||  Ex_B  (pg_inter)
      t3:  AG_B  (pg_intra_b)

  The "||" pairs use distinct process groups, so they execute on distinct
  NCCL streams and can run concurrently on the GPU / network. This hides
  most of the inter-node exchange latency behind the intra-node stages.

Usage:
  torchrun --nproc_per_node=8 --nnodes=1 benchmark_both.py --mode naive
  torchrun --nproc_per_node=8 --nnodes=2 ... benchmark_both.py --mode naive
"""

import os
import torch
import torch.distributed as dist


# Pipeline only for sizes at/above this threshold (in elements of `dtype`).
# Below this, the extra PG switching / multiple batch calls cost more than
# the overlap saves.  32 MB bf16 = 16M elements.
_PIPELINE_MIN_ELEMENTS = 16 * 1024 * 1024  # = 32 MB at bfloat16


class NaiveAllReduce:
    """Naive all-to-all allreduce with pre-allocated recv buffers.

    * Single node     : flat RS+AG.
    * Multi-node small: serial hierarchical RS / inter-exchange / AG.
    * Multi-node large: pipelined hierarchical on two intra-node PGs.
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
        self._intra_group_a = None     # primary intra-node PG
        self._intra_group_b = None     # secondary intra-node PG (pipelining)
        self._inter_group = None
        self._intra_info_a = None      # cached (rank, ws, global_ranks)
        self._intra_info_b = None
        self._inter_info = None
        self._inter_peer = -1          # global rank of sole peer (num_nodes==2)

        if self.num_nodes > 1:
            # Primary intra group
            for nid in range(self.num_nodes):
                ranks = list(range(nid * self.local_ws, (nid + 1) * self.local_ws))
                g = dist.new_group(ranks=ranks)
                if nid == self.node_id:
                    self._intra_group_a = g
            # Secondary intra group (independent NCCL comm for pipelining)
            for nid in range(self.num_nodes):
                ranks = list(range(nid * self.local_ws, (nid + 1) * self.local_ws))
                g = dist.new_group(ranks=ranks)
                if nid == self.node_id:
                    self._intra_group_b = g
            # Inter-node group
            for lr in range(self.local_ws):
                ranks = [nid * self.local_ws + lr for nid in range(self.num_nodes)]
                g = dist.new_group(ranks=ranks)
                if lr == self.local_rank:
                    self._inter_group = g

            self._intra_info_a = (dist.get_rank(self._intra_group_a),
                                  dist.get_world_size(self._intra_group_a),
                                  dist.get_process_group_ranks(self._intra_group_a))
            self._intra_info_b = (dist.get_rank(self._intra_group_b),
                                  dist.get_world_size(self._intra_group_b),
                                  dist.get_process_group_ranks(self._intra_group_b))
            self._inter_info = (dist.get_rank(self._inter_group),
                                dist.get_world_size(self._inter_group),
                                dist.get_process_group_ranks(self._inter_group))
            if self.num_nodes == 2:
                ir_rank, _, ir_globals = self._inter_info
                self._inter_peer = ir_globals[1 - ir_rank]

        # ── Pre-allocate recv buffers ──
        # Serial path:   one buffer sized for intra RS of the full tensor
        #                = (local_ws-1) * (n/local_ws)  ≈ 7n/8 for local_ws=8
        # Pipelined path: two per-track buffers, each sized for intra RS of
        #                HALF the tensor = (local_ws-1) * (n/2/local_ws) = 7n/16
        #                total across tracks = 7n/8  — same as serial.
        # We lay out both tracks in a single allocation and slice.
        if self.num_nodes > 1:
            intra_C = max_elements // self.local_ws
            full_buf = (self.local_ws - 1) * intra_C
        else:
            full_buf = (self.world_size - 1) * (max_elements // self.world_size)
        self.recv_buf = torch.empty(max(1, full_buf), dtype=dtype, device=device)

    # ─────────────────────── group-info dispatch ───────────────────────

    def _group_info(self, group):
        if group is None:
            return self.rank, self.world_size, list(range(self.world_size))
        if group is self._intra_group_a:
            return self._intra_info_a
        if group is self._intra_group_b:
            return self._intra_info_b
        if group is self._inter_group:
            return self._inter_info
        return (dist.get_rank(group),
                dist.get_world_size(group),
                dist.get_process_group_ranks(group))

    # ─────────────────────── serial primitives ───────────────────────

    def _rs(self, tensor, group, recv_buf=None):
        rank, ws, global_ranks = self._group_info(group)
        if ws <= 1:
            return
        n = tensor.numel()
        C = n // ws
        npeers = ws - 1
        if recv_buf is None:
            recv_buf = self.recv_buf
        recv = recv_buf[:npeers * C]

        ops = []
        buf_i = 0
        for peer in range(ws):
            if peer == rank:
                continue
            pg = global_ranks[peer]
            ops.append(dist.P2POp(dist.isend,
                                  tensor[peer * C:(peer + 1) * C],
                                  pg, group=group))
            ops.append(dist.P2POp(dist.irecv,
                                  recv[buf_i * C:(buf_i + 1) * C],
                                  pg, group=group))
            buf_i += 1
        for r in dist.batch_isend_irecv(ops):
            r.wait()

        my = rank * C
        if npeers == 1:
            tensor[my:my + C].add_(recv[:C])
        else:
            tensor[my:my + C].add_(recv.view(npeers, C).sum(dim=0))

    def _ag(self, tensor, group):
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
            pg = global_ranks[peer]
            ops.append(dist.P2POp(dist.isend, my_shard, pg, group=group))
            ops.append(dist.P2POp(dist.irecv,
                                  tensor[peer * C:(peer + 1) * C],
                                  pg, group=group))
        for r in dist.batch_isend_irecv(ops):
            r.wait()

    def _flat_ar(self, tensor, group):
        self._rs(tensor, group)
        self._ag(tensor, group)

    def _inter_exchange_2nodes(self, shard, recv_buf=None):
        """Stage 2 for num_nodes==2: send + recv + add. Split into 2 halves
        issued in a single batch call to keep per-message size moderate."""
        C = shard.numel()
        if recv_buf is None:
            recv_buf = self.recv_buf
        recv = recv_buf[:C]
        half = C // 2
        peer = self._inter_peer
        grp = self._inter_group
        if half == 0:
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

    # ─────────────────────── pipelined primitives ───────────────────────
    # These split _rs / _ag / _inter_exchange into "issue" (non-blocking —
    # returns a finalize closure) and "finalize" (waits + does the post-comm
    # add/sum) so we can interleave work from multiple tracks.

    def _rs_issue(self, tensor, group, info, recv_buf):
        rank, ws, global_ranks = info
        n = tensor.numel()
        C = n // ws
        npeers = ws - 1
        recv = recv_buf[:npeers * C]

        ops = []
        buf_i = 0
        for peer in range(ws):
            if peer == rank:
                continue
            pg = global_ranks[peer]
            ops.append(dist.P2POp(dist.isend,
                                  tensor[peer * C:(peer + 1) * C],
                                  pg, group=group))
            ops.append(dist.P2POp(dist.irecv,
                                  recv[buf_i * C:(buf_i + 1) * C],
                                  pg, group=group))
            buf_i += 1
        reqs = dist.batch_isend_irecv(ops)
        my = rank * C

        def finalize():
            for r in reqs:
                r.wait()
            if npeers == 1:
                tensor[my:my + C].add_(recv[:C])
            else:
                tensor[my:my + C].add_(recv.view(npeers, C).sum(dim=0))
        return finalize

    def _ag_issue(self, tensor, group, info):
        rank, ws, global_ranks = info
        n = tensor.numel()
        C = n // ws
        my = rank * C
        my_shard = tensor[my:my + C]

        ops = []
        for peer in range(ws):
            if peer == rank:
                continue
            pg = global_ranks[peer]
            ops.append(dist.P2POp(dist.isend, my_shard, pg, group=group))
            ops.append(dist.P2POp(dist.irecv,
                                  tensor[peer * C:(peer + 1) * C],
                                  pg, group=group))
        reqs = dist.batch_isend_irecv(ops)

        def finalize():
            for r in reqs:
                r.wait()
        return finalize

    def _ex_issue_2nodes(self, shard, recv_buf):
        C = shard.numel()
        recv = recv_buf[:C]
        half = C // 2
        peer = self._inter_peer
        grp = self._inter_group
        if half == 0:
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
        reqs = dist.batch_isend_irecv(ops)

        def finalize():
            for r in reqs:
                r.wait()
            shard.add_(recv[:C])
        return finalize

    # ─────────────────────── main entry ───────────────────────

    def __call__(self, input_tensor):
        if self.num_nodes <= 1:
            self._flat_ar(input_tensor, None)
            return input_tensor

        n = input_tensor.numel()
        # Pipelined path for large tensors (only for num_nodes==2 and a
        # cleanly 2-way splittable size).
        if (n >= _PIPELINE_MIN_ELEMENTS
                and self.num_nodes == 2
                and (n // 2) % self.local_ws == 0
                and (n // 2) // self.local_ws > 0):
            self._hierarchical_pipelined_2nodes(input_tensor)
            return input_tensor

        # Serial hierarchical (existing)
        C_node = n // self.local_ws
        self._rs(input_tensor, self._intra_group_a)
        shard = input_tensor[self.local_rank * C_node:
                             (self.local_rank + 1) * C_node]
        if self.num_nodes == 2:
            self._inter_exchange_2nodes(shard)
        else:
            self._flat_ar(shard, self._inter_group)
        self._ag(input_tensor, self._intra_group_a)
        return input_tensor

    # ─────────────────────── pipelined hierarchical ───────────────────────

    def _hierarchical_pipelined_2nodes(self, input_tensor):
        """Overlap inter-node exchange with intra-node RS/AG of the other
        half, using two independent intra-node process groups.

        Timeline (arrows = dependency inside a track, "||" = concurrent):

            RS_A  →  Ex_A  →  AG_A
                 ↘       ↘
            RS_B  →  Ex_B  →  AG_B

            t0:  issue RS_A (pg_intra_a)
            t1:  issue RS_B (pg_intra_b)   [concurrent with RS_A]
            t2:  wait RS_A, issue Ex_A     [Ex_A runs while RS_B finishes]
            t3:  wait RS_B
            t4:  wait Ex_A, issue AG_A and Ex_B [AG_A and Ex_B concurrent]
            t5:  wait AG_A, wait Ex_B, issue AG_B
            t6:  wait AG_B
        """
        n = input_tensor.numel()
        local_ws = self.local_ws
        lr = self.local_rank

        # Split exactly in half (caller guaranteed (n/2) % local_ws == 0)
        half_n = n // 2
        A = input_tensor[:half_n]
        B = input_tensor[half_n:]

        half_C = half_n // local_ws         # per-GPU shard size for each half
        shard_A = A[lr * half_C:(lr + 1) * half_C]
        shard_B = B[lr * half_C:(lr + 1) * half_C]

        # Recv buffers: split the pre-allocated recv_buf into two halves,
        # one per track. Each track needs (local_ws-1)*half_C for its intra RS
        # and only half_C for its inter exchange (reused).
        intra_recv_size = (local_ws - 1) * half_C
        track_size = intra_recv_size                         # dominates
        recv_a = self.recv_buf[:track_size]
        recv_b = self.recv_buf[track_size:2 * track_size]

        pg_a = self._intra_group_a
        pg_b = self._intra_group_b
        info_a = self._intra_info_a
        info_b = self._intra_info_b

        # t0 / t1 — issue both intra RSes concurrently
        fin_rs_a = self._rs_issue(A, pg_a, info_a, recv_a)
        fin_rs_b = self._rs_issue(B, pg_b, info_b, recv_b)

        # t2 — wait RS_A, launch Ex_A (Ex_A will overlap with tail of RS_B)
        fin_rs_a()
        fin_ex_a = self._ex_issue_2nodes(shard_A, recv_a)

        # t3 — wait RS_B. Ex_A is still in flight on inter PG; RS_B was on
        #      pg_b (independent stream) so this wait does not block Ex_A.
        fin_rs_b()

        # t4 — wait Ex_A, then issue AG_A and Ex_B concurrently
        fin_ex_a()
        fin_ag_a = self._ag_issue(A, pg_a, info_a)
        fin_ex_b = self._ex_issue_2nodes(shard_B, recv_b)

        # t5 — wait AG_A and Ex_B (order doesn't matter), then issue AG_B
        fin_ag_a()
        fin_ex_b()
        fin_ag_b = self._ag_issue(B, pg_b, info_b)

        # t6 — final wait
        fin_ag_b()


_naive_instance = None


def naive_sendrecv_allreduce(input_tensor):
    global _naive_instance
    n = input_tensor.numel()
    ws = dist.get_world_size()
    local_ws = int(os.environ.get("LOCAL_WORLD_SIZE", ws))

    if ws // local_ws > 1:
        intra_C = n // local_ws
        required = (local_ws - 1) * intra_C
    else:
        required = (ws - 1) * (n // ws)

    if _naive_instance is None or _naive_instance.recv_buf.numel() < required:
        _naive_instance = NaiveAllReduce(n, input_tensor.dtype, input_tensor.device)
    return _naive_instance(input_tensor)
