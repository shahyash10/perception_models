import mmh3
import argparse
import copy
import os
import json
import random
import re
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
from apps.plm.generate import (
    PackedCausalTransformerGenerator,
    PackedCausalTransformerGeneratorArgs,
    load_consolidated_model_and_tokenizer,
)
from torch.utils.data import Dataset, DataLoader

from core.transforms.image_transform import get_image_transform
import math
from apps.plm.cambrian_eval_utils import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,conv_llama_3, tokenizer_image_token

def hash_image(image):
    image = image.resize((256, 256))
    image_bytes = image.tobytes()
    hash_value = mmh3.hash(image_bytes)
    hex_hash = format(hash_value, '08x')
    return hex_hash


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(lst / n)  # integer division
    return [[i,i+chunk_size-1] for i in range(0, lst, chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def process(line, args, tokenizer, image_processor, model_config):
    if line["hint"] != "nan":
        qs = line["hint"] + "\n" + line["question"] + f" Options:"
    else:
        qs = line["question"] + f" Options:"
    for keys in ["A", "B", "C", "D"]:
        if line[keys] != "nan":
            qs += (f"\n{keys}. "+line[keys])
    
    qs += f"\n{args.question_extension}"
    if line["image"] is not None:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    # Use the conv_llama_3 template
    conv = conv_llama_3.copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)

    structured_conversation = [
        {"role": conv.roles[0], "content": qs},
        {"role": conv.roles[1], "content": None}
    ]

    image_hash = ""
    if line["image"] is None:
        image = None
        image_size = None
        image_tensor = None
    else:
        image = line["image"].convert('RGB')
        image_hash = hash_image(copy.deepcopy(image))
        image_size = [image.size]
        image_tensor, _ = image_processor(image)

    input_ids = tokenizer_image_token(structured_conversation, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    return input_ids, image_tensor, image_size, image_hash, structured_conversation


def eval_model(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Model
    model, tokenizer, config = load_consolidated_model_and_tokenizer(
        args.model_path,
        tokenizer_path="/home/yashs/projects/perception_models/facebook/Perception-LM-1B/tokenizer.model"
    )

    # Wrap the model with PackedCausalTransformerGenerator
    gen_cfg = PackedCausalTransformerGeneratorArgs(
        temperature=args.temperature,
        top_p=args.top_p,
        max_gen_len=args.max_new_tokens,
    )
    generator = PackedCausalTransformerGenerator(gen_cfg, model, tokenizer)

    # Image processor
    image_processor = get_image_transform(
        vision_input_type=config.data.vision_input_type,
        image_res=config.model.vision_model.image_size,
        max_num_tiles=config.data.max_num_tiles,
    )    
    questions = load_dataset("lmms-lab/MMBench_CN", "default", split="dev")
    
    answers_file = os.path.expanduser(args.answers_file)
    if not answers_file.endswith(".jsonl"):
        raise ValueError("Answers file must be a jsonl file")
    
    basename = os.path.basename(answers_file)
    basename = os.path.splitext(basename)[0]
    answers_dir = os.path.dirname(answers_file)
    chunk_fname = f"{basename}_{args.chunk_idx}.jsonl"
    chunk_file = os.path.join(answers_dir, chunk_fname)
    os.makedirs(os.path.dirname(chunk_file), exist_ok=True)

    idx = -1
    valid_chunk = get_chunk(len(questions), args.num_chunks, args.chunk_idx)
    print(valid_chunk)
    
    with open(chunk_file, "w") as ans_file:
        for line in tqdm(questions, total=len(questions)):
            idx = idx+1
            if idx<valid_chunk[0] or idx>valid_chunk[1]:
                continue
            
            input_ids, image_tensor, image_sizes, image_hash, prompt = process(line, args, tokenizer, image_processor, model.config)
            gt_answer = line["answer"]
            category = line["category"]
            # l2_category = line["l2-category"]
            input_ids = input_ids.to(device='cuda', non_blocking=True)
            with torch.inference_mode():
                generated_text = generator.generate( [(prompt[0]["content"], image_tensor)] if image_tensor is not None else [(prompt[0]["content"], None)])[0]
            if isinstance(generated_text, list):
                generated_text = generated_text[0]

            source_id = (line["question"]+ " " + image_hash + " " + line["source"])
            ans_file.write(json.dumps({"index": line["index"],
                                    "question": line["question"],
                                    "prediction": generated_text,
                                    "gt_answer": gt_answer,
                                    "A":line["A"],
                                    "B":line["B"],
                                    "C":line["C"],
                                    "D":line["D"],
                                    "source_id": source_id,
                                    "model_id": args.model_path,
                                    "category": category}) + "\n")
            ans_file.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/fsx-checkpoints/yashs/plm/plm_1b_cambrian7M_stage2_1024_baseline1/checkpoints/0000007000/")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--answers_file", type=str, default="./answers/answers.jsonl")
    parser.add_argument("--question_extension", type=str, default="请直接回答选项字母。")
    parser.add_argument("--conv_mode", type=str, default="llama_3")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    eval_model(args)