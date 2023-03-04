# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json
import random

from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama.llama import ModelArgs, Transformer, Tokenizer, LLaMA

def setup_model_parallel() -> Tuple[int, int]:
    local_rank = 0#int(os.environ.get("LOCAL_RANK", -1))
    world_size = 1#int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(random.randrange(1,65,536))
    return local_rank, world_size


def load_model(ckpt_dir: str, tokenizer_path: str, local_rank: int, world_size: int) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert (
        world_size == len(checkpoints)
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(max_seq_len=2048, max_batch_size=1, **params)
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def generate_model(generator, temperature_: float, top_p_: float, max_output: int):

    print(temperature_)
    print(top_p_)
    print(max_output)

    prompt =  open('drive/MyDrive/llama/prompt.txt','r').read()
    print(prompt)
    print("\n==================================\n")
    
    prompts = [prompt]
    results = generator.generate(prompts, max_gen_len=max_output, temperature=temperature_, top_p=top_p_)

    for result in results:
        print(result)
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
