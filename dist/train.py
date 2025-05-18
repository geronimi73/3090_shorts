import torch 
import wandb 
import contextlib
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler, DataLoader
from torchvision import datasets, transforms

from types import SimpleNamespace
from model import Net

def dist_init():
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(dist.get_rank())

def dist_finish():
    dist.destroy_process_group()

def dist_gather(o):
    o_all = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(o_all, o)
    return o_all

def get_world_size(): return dist.get_world_size()
def is_master(): return dist.get_rank() == 0

def log_init(config):
    if is_master():
        wandb.init(
            project = "DDP-minimal", 
            name = f"GLOB-BS-{config.bs * config.gas * get_world_size()}_BS-{config.bs}_GA-{config.gas}_GPUS-{get_world_size()}"
        ).log_code(".", include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb") or path.endswith(".json"))

def log_finish():
    if is_master():
        wandb.finish()

def log(step, loss):
    loss_all = dist_gather(loss)
    loss_avg = sum(loss_all) / len(loss_all)

    print(f"Step {step} Loss {loss_avg}")

    if wandb.run is not None and is_master():
        wandb.log({"step": step, "loss_train": loss_avg})    

def get_dataloaders(bs_train=32, bs_test=32):
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataloader_train = DataLoader(
        dataset_train,
        sampler = DistributedSampler(dataset_train, shuffle=False),
        batch_size = bs_train,
    )

    dataset_test = datasets.MNIST('./data', train=False, download=True, transform=transform)
    dataloader_test = DataLoader(
        dataset_test,
        batch_size = bs_test,
        shuffle = False,
    )

    return dataloader_train, dataloader_test


def train(train_config, device="cuda"):
    # load and wrap model
    model = Net().to(device)
    model = DistributedDataParallel(model, device_ids=[dist.get_rank()])

    dataloader_train, dataloader_test = get_dataloaders(bs_train=train_config.bs)
    optimizer = torch.optim.AdamW(model.parameters(), train_config.lr)

    step, batch_loss = 0, 0

    for batch_idx, (data, target) in enumerate(dataloader_train):

        # step optimizer if last micro batch of batch or last batch in dataloader
        if ( (batch_idx+1) % train_config.gas == 0 
            or batch_idx + 1 == len(dataloader_train) ):
            step_optimizer = True
        else:
            step_optimizer = False
        
        with model.no_sync() if not step_optimizer else contextlib.nullcontext():
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
            if step % train_config.log_interval == 0: 
                log(step, batch_loss)

            batch_loss = 0
            step += 1
        

def main():
    torch.manual_seed(42)

    train_config = SimpleNamespace(
        lr = 0.0001,
        # global batch size will be (bs * gas * num_GPUs)
        bs = 8,
        gas = 2,       # =gradient accumulation steps
        epochs = 3,
        log_interval = 50,
    )

    dist_init()
    log_init(train_config)

    train(train_config)

    log_finish()
    dist_finish()

if __name__ == '__main__':
    main()