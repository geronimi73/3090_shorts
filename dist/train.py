import torch 
import wandb 
import torch.nn.functional as F
import torch.distributed as dist
from torchvision import datasets, transforms
from torch.utils.data import DistributedSampler, DataLoader
from types import SimpleNamespace
# REMOVE ME!
from transformers import set_seed
from model import Net

def dist_init():
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(get_rank())

def dist_destroy():
    dist.destroy_process_group()

def get_rank():
    return dist.get_rank()

def get_world_size():
    return dist.get_world_size()

def is_master():
    return get_rank() == 0

def log_init(config):
    wandb.init(
        project = "DDP-minimal", 
        name = f"GLOB-BS-{config.bs * config.gas}_BS-{config.bs}_GA-{config.gas}"
    ).log_code(".", include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb") or path.endswith(".json"))

def log_finish():
    wandb.finish()

def log(step, loss):
    print(f"Step {step} Loss {loss}")
    wandb.log({"step": step, "loss_train": loss})    

def get_dataloaders(bs_train=32, bs_test=32):
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataloader_train = DataLoader(
        dataset_train,
        sampler = DistributedSampler(
            dataset_train, 
            # remove me
            rank=get_rank(),
            num_replicas=get_world_size(),            
            shuffle=False
        ),
        batch_size = bs_train,
        shuffle = False,
    )

    dataset_test = datasets.MNIST('./data', train=False, download=True, transform=transform)
    dataloader_test = DataLoader(
        dataset_test,
        batch_size = bs_test,
        shuffle = False,
    )

    return dataloader_train, dataloader_test

dist_init()

device = "cuda"
train_config = SimpleNamespace(
    log_interval = 50,
    lr = 0.0001,
    bs = 32,
    gas = 1,
)

micro_batch_size = 8
train_config.gas = train_config.bs // micro_batch_size
train_config.bs = train_config.bs // train_config.gas

set_seed(42)


dataloader_train, dataloader_test = get_dataloaders(bs_train=train_config.bs)
model = Net().to(device)
optimizer = torch.optim.AdamW(model.parameters(), train_config.lr)

log_init(train_config)

step, batch_loss = 0, 0

for batch_idx, (data, target) in enumerate(dataloader_train):

    # step optimizer if last micro batch of batch or last batch in dataloader
    if ( (batch_idx+1) % train_config.gas == 0 
        or batch_idx + 1 == len(dataloader_train) ):
        step_optimizer = True
    else:
        step_optimizer = False
    
    output = model(data.to(device))
    loss = F.nll_loss(output, target.to(device))
    # divide loss by number of gradient acc. steps
    loss = loss / train_config.gas if train_config.gas > 1 else loss
    loss.backward()

    # add up loss of microbatches 
    batch_loss += loss.item()

    if step_optimizer:
        optimizer.step()
        optimizer.zero_grad()

        # Log step
        if step % train_config.log_interval == 0: log(step, batch_loss)

        batch_loss = 0
        step += 1
        
log_finish()
dist_destroy()