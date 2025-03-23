# Based on multiprocessing example from
# https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html

import time
import contextlib
import torch
import torch.nn as nn
import torch.distributed as dist
import torchvision.transforms as transforms
import wandb
from torchvision.datasets import MNIST
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, RandomSampler
from statistics import mean

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

def ddp_gather(o):
    o_all = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(o_all, o)
    return o_all


def train():
    dist.init_process_group(backend='nccl')

    seed = 42
    torch.manual_seed(seed)

    is_master = dist.get_rank() == 0  
    world_size = dist.get_world_size()
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)

    num_epochs = 3
    lr = 1e-4
    log_steps = 20

    model = ConvNet().cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr)

    model = DistributedDataParallel(model, device_ids=[local_rank])

    train_dataset = MNIST(
        root = './data', train = True, transform = transforms.ToTensor(), download = True
    )

    batch_size = 256
    gradient_accumulation_steps = 1

    train_sampler = DistributedSampler(
        train_dataset, shuffle = False, seed = seed
    )
    train_loader = DataLoader(
        dataset = train_dataset, batch_size = batch_size, sampler = train_sampler,
        shuffle = False, 
        num_workers = 0, pin_memory = True,
    )

    if is_master: wandb.init(project="DDP_GRAD-ACC", name=f"GLOB-BS-{batch_size*gradient_accumulation_steps*world_size}_BS-{batch_size}_GAS-{gradient_accumulation_steps}_{world_size}-GPUs")

    step = 0
    sample_count = 0
    last_step_time = time.time()
    loss_cum = 0

    for epoch in range(num_epochs):
        accumulation_step = 0

        train_sampler.set_epoch(epoch)

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            sample_count += len(images)
            accumulation_step += 1

            if gradient_accumulation_steps > 1 and accumulation_step % gradient_accumulation_steps != 0:
                context = model.no_sync()
            else:
                context = contextlib.nullcontext()

            with context:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss = loss / gradient_accumulation_steps
                loss.backward()
                loss_cum += loss.cpu().detach().item() 

            if ((batch_idx+1) % gradient_accumulation_steps == 0) or ((batch_idx+1)==len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()

                step += 1
                loss_cum = mean(ddp_gather(loss_cum))

                if is_master and step % log_steps == 0:
                    metrics = dict(
                        step = step,
                        loss = loss_cum,
                        sample_count = sample_count * world_size,
                        # sample_tp = sample_tp,
                    )
                    wandb.log(metrics)
                    print(", ".join([f"{k}: {round(metrics[k],2) if isinstance(metrics[k], float) else metrics[k]}" for k in metrics]))
                loss_cum = 0

    if is_master: 
        wandb.finish()

        images, labels = next(iter(train_loader))
        with torch.no_grad():
            outputs = model(images.cuda())
        loss = criterion(outputs, labels.cuda())

        print(f"final loss: {loss.item()}")


    dist.destroy_process_group()

if __name__ == '__main__':
    train()
