from types import SimpleNamespace
import wandb 
import contextlib
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler, DataLoader
from torchvision import datasets, transforms

# source: https://github.com/pytorch/examples/blob/main/mnist/main.py
# dropouts disable for reproducibility
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # self.dropout1 = nn.Dropout(0.25)
        # self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def dist_init():
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(dist.get_rank())

def dist_finish():
    dist.destroy_process_group()

def dist_gather(o):
    o_all = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(o_all, o)
    return o_all

def is_master(): 
    return dist.get_rank() == 0

def get_dataloaders(bs_train=32, bs_test=32):
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataloader_train = DataLoader(
        dataset_train,
        sampler = DistributedSampler(dataset_train, shuffle=True),
        batch_size = bs_train,
    )

    dataset_test = datasets.MNIST('./data', train=False, download=True, transform=transform)
    dataloader_test = DataLoader(
        dataset_test,
        sampler = DistributedSampler(dataset_test, shuffle=False),
        batch_size = bs_test,
    )

    return dataloader_train, dataloader_test

def log_init(config):
    if is_master():
        wandb.init(
            project = "DDP-minimal", 
            name = "GLOB-BS-{bs_glob}_BS-{bs_local}_GA-{gas}_GPUS-{num_gpus}".format(
                bs_glob = config.bs * config.gas * dist.get_world_size(),
                bs_local = config.bs,
                gas = config.gas,
                num_gpus = dist.get_world_size(),
            )
        ).log_code(".", include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb") or path.endswith(".json"))

def log_finish():
    if is_master():
        wandb.finish()

def log(epoch, step, loss):
    loss_all = dist_gather(loss)
    loss_avg = sum(loss_all) / len(loss_all)

    if is_master():
        print(f"Step {step}, epoch {epoch}, loss {loss_avg}")

        if wandb.run is not None:
            wandb.log({"step": step, "loss_train": loss_avg})    

def test(step, model, dataloader_test, device="cuda"):
    model.eval()
    test_loss, correct, num_samples = 0, 0, 0

    for data, target in dataloader_test:
        with torch.no_grad():
            # unwrap model for evals
            output = model.module(data.to(device))
        test_loss += F.nll_loss(output, target.to(device), reduction='sum').item()  # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True).cpu()  # get the index of the max log-probability
        correct += pred.eq(target.cpu().view_as(pred)).sum().item()
        num_samples += target.shape[0]

    # gather results from all GPUs
    num_samples = sum([ns for ns in dist_gather(num_samples)])
    correct = sum([c for c in dist_gather(correct)])
    test_loss = sum([tl for tl in dist_gather(test_loss)])

    # log on master only
    if is_master():
        accuracy = 100. * correct / num_samples
        test_loss /= num_samples

        print(f"Test set: Loss {test_loss:.2f}, accuracy {accuracy:.2f}")

        if wandb.run is not None:
            wandb.log({"step": step, "accuracy": accuracy, "loss_test": test_loss})    

    model.train()


def train(train_config, device="cuda"):
    # load and wrap model
    model = Net().to(device)
    model = DistributedDataParallel(model)

    dataloader_train, dataloader_test = get_dataloaders(bs_train=train_config.bs)
    optimizer = torch.optim.AdamW(model.parameters(), train_config.lr)

    step, batch_loss = 0, 0

    for epoch in range(train_config.epochs):
        dataloader_train.sampler.set_epoch(epoch)

        for batch_idx, (data, target) in enumerate(dataloader_train):

            # step optimizer if last micro batch of batch OR last batch in dataloader
            if ( (batch_idx+1) % train_config.gas == 0 
                or batch_idx + 1 == len(dataloader_train) ):
                step_optimizer = True
            else:
                step_optimizer = False
            
            # prevent DDP gradient sync if not stepping
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
                    log(epoch, step, batch_loss)

                # Eval step
                if step % train_config.eval_interval == 0: 
                    test(step, model, dataloader_test)

                batch_loss = 0
                step += 1
            

def main():
    torch.manual_seed(42)

    train_config = SimpleNamespace(
        lr = 0.0001,
        # global batch size will be (bs * gas * num_GPUs)
        bs = 16,
        gas = 1,       # =gradient accumulation steps
        epochs = 3,
        log_interval = 50,
        eval_interval = 200,
    )

    dist_init()
    log_init(train_config)

    train(train_config)

    log_finish()
    dist_finish()

if __name__ == '__main__':
    main()