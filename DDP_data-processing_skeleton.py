import torch
import torch.distributed as dist
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

def process(rank, is_master, world_size, batch_size=8):
	print(f"Hello from rank {rank}")

	# Create a dummy dataset
	# ds = load_dataset(..)
	ds = Dataset.from_list(
		[{"data": f"data {i}"} for i in range(100)]
	)

	# Distributed data loader without shuffling
	def collate_fn(items):
		data_list = [item["data"] for item in items]
		return data_list
	sampler = DistributedSampler(ds, shuffle=False)
	dataloader = DataLoader(ds, sampler=sampler, collate_fn=collate_fn, batch_size=batch_size)

	# Process your data
	for batch in tqdm(dataloader, desc=f"rank {rank}"):
		pass


if __name__ == '__main__':
	dist.init_process_group(backend='nccl')

	seed = 42
	torch.manual_seed(seed)

	world_size = dist.get_world_size()
	rank = dist.get_rank()
	is_master = rank == 0  
	torch.cuda.set_device(rank)

	process(rank, is_master, world_size)

	dist.destroy_process_group()
