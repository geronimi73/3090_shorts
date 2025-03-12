import torch
import torch.distributed as dist
import torchvision.transforms as T
import requests, json, os, time
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText
 
from utils_captioning import batch_caption_qwenvlm, batch_caption_smolvlm2, batch_caption_moondream

def load_imagenet_labels():
    raw_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    response = requests.get(raw_url)
    imagenet_labels = json.loads(response.text)
    return imagenet_labels

def process(rank, is_master, world_size):
    print(f"Hello from rank {rank}")

    ## Load VLM
    device = "cuda"
    dtype = torch.bfloat16
    debug = True
    batch_size = 8
    num_samples = 200
    captioningf = batch_caption_qwenvlm

    # Suppress tokenizers warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    ## Load dataset and class names
    ds = load_dataset("ILSVRC/imagenet-1k", streaming=True, trust_remote_code=True)

    # Subsample train split and convert IterableDataset to Dataset (via list)  
    ds = ds["train"].take(num_samples + 100)
    ds = Dataset.from_list(list(ds), features=ds.features)

    # Load ImageNet1k labels
    in_labels = load_imagenet_labels()

    ## Prepare dataloader
    def collate_fn(items):
        # 1) resize image to save VRAM
        # 2) extract classname. d["label"] is a number (e.g. 23)  -> string ('vulture')
        return [
            [T.CenterCrop(256)(T.Resize(256)(i["image"])) for i in items],
            [in_labels[i["label"]] for i in items],
        ]
        
    sampler = DistributedSampler(ds, shuffle=False)
    dataloader = DataLoader(
        ds, 
        sampler=sampler,
        collate_fn=collate_fn, 
        batch_size=batch_size,     # 8 samples per batch, maximum which fits on a 24GB VRAM GPU 
        num_workers=2,    # Use 2 threads for prefetching
        prefetch_factor=10  # Each thread will prefetch 10 samples
    )

    prompt_template = """The image shows a {class_name}. Please come up 
with a short image caption, list and describe the main objects shown 
in the image. Keep the caption short, one sentence only."""

    ## mock caption to load model
    img = ds[0]["image"]
    captioningf([img],["What is this??"])

    # wait for everyone
    torch.distributed.barrier()

    ## Sample processing loop
    time_start = time.time()
    gpu_time = 0
    samples = 0 

    for images, class_names in tqdm(dataloader, f"GPU{rank}"):
        # insert class name into prompts
        prompts = [
            prompt_template.format(class_name=class_names[i])
            for i in range(len(images))
        ]
        gpu_start = time.time()
        captions = captioningf(images, prompts)
        gpu_time += time.time() - gpu_start
        # for c in captions:
        #     print(c)
        samples += batch_size * world_size
        if samples > num_samples: break

    # wait for everyone
    torch.distributed.barrier()

    if is_master:
        time_total = time.time()-time_start
        print(f"samples per second: {samples/time_total:.2f}, total samples: {samples}")
        print(f"total time {time_total:.2f} seconds, GPU time: {gpu_time:.2f} seconds, GPU time {gpu_time/time_total*100:.2f}%")

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
