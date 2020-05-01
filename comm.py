#!/bin/python3

import os
import torch

rank = int(os.environ['SLURM_PROCID'])
local_rank = int(os.environ['SLURM_LOCALID'])
ip_addr = os.environ['MASTER_ADDR']
size = int(os.environ['SLURM_NTASKS'])
print("rank : %d" %rank)
print("size : %d" %size)
backend = 'nccl'
method = 'tcp://' + ip_addr + ":22233"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.distributed.init_process_group(
	backend, world_size=size, rank=rank, init_method=method)


if (rank == 1):
	send_x = torch.ones([1,5]).cuda(device)
	torch.distributed.broadcast(send_x, rank)

else:
	recv_x = torch.zeros([1,5]).cuda(device)
	print("local tensor ============== ")
	print(recv_x)
	torch.distributed.reduce(recv_x, 0)
	print("recv tensor =============== ")
	print(recv_x)
 


