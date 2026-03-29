"""
Test isend/irecv consistency between 2 nodes.

Node 0: torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr=172.31.12.80 --master_port=29500 test_isend_irecv.py
Node 1: torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr=172.31.12.80 --master_port=29500 test_isend_irecv.py
"""
import torch
import torch.distributed as dist

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
peer = 1 - rank
device = torch.device("cuda:0")
torch.cuda.set_device(device)

sizes = [1024, 1 << 20, 1 << 22, 1 << 24]

for size in sizes:
    for trial in range(3):
        # Each rank sends a tensor filled with (rank+1)*10 + trial
        fill_val = (rank + 1) * 10 + trial
        expected_val = (peer + 1) * 10 + trial

        send_buf = torch.full([size], fill_val, dtype=torch.uint8, device=device)
        recv_buf = torch.zeros([size], dtype=torch.uint8, device=device)
        torch.cuda.synchronize()

        ops = [
            dist.P2POp(dist.isend, send_buf, peer),
            dist.P2POp(dist.irecv, recv_buf, peer),
        ]
        reqs = dist.batch_isend_irecv(ops)
        for r in reqs:
            r.wait()
        torch.cuda.synchronize()

        n_correct = (recv_buf == expected_val).sum().item()
        n_wrong = size - n_correct
        status = "OK" if n_wrong == 0 else "FAIL"

        print(f"[rank{rank}] size={size:>10} trial={trial} sent={fill_val} expect={expected_val} "
              f"correct={n_correct} wrong={n_wrong} ({n_wrong/size*100:.2f}%) {status}", flush=True)

        if n_wrong > 0:
            wrong_idx = (recv_buf != expected_val).nonzero(as_tuple=True)[0]
            print(f"  first 10 wrong idx: {wrong_idx[:10].tolist()}", flush=True)
            print(f"  first 10 wrong val: {recv_buf[wrong_idx[:10]].tolist()}", flush=True)

dist.barrier()
if rank == 0:
    print("\nDone!", flush=True)
dist.destroy_process_group()
