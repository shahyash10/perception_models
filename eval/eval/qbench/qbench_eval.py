import argparse

import os
import json
import random
import re
import tarfile
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
from apps.plm.generate import (
    PackedCausalTransformerGenerator,
    PackedCausalTransformerGeneratorArgs,
    load_consolidated_model_and_tokenizer,
)

from core.transforms.image_transform import get_image_transform
import math
from apps.plm.cambrian_eval_utils import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,conv_llama_3, tokenizer_image_token
from PIL import Image
from huggingface_hub import hf_hub_download

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(lst / n)  # integer division
    return [[i,i+chunk_size-1] for i in range(0, lst, chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def process(line, args, tokenizer, image_processor, model_config, images):
    qs = line["question"] + " Options:"
    options = line["candidates"]
    for i in range(len(options)):
        option = chr(ord('A')+i)
        text = options[i]
        qs += f"\n{option}. {text}"

    qs += f"\n{args.question_extension}"

    image_path = images[line["img_path"]]
    input_image = Image.open(image_path).convert('RGB')

    if input_image is not None:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    # Use the conv_llama_3 template
    conv = conv_llama_3.copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)

    structured_conversation = [
        {"role": conv.roles[0], "content": qs},
        {"role": conv.roles[1], "content": None}
    ]
    if input_image is None:
        image = None
        image_size = None
        image_tensor = None
    else:
        image = input_image
        image_size = [image.size]
        image_tensor, _ = image_processor(image)

    input_ids = tokenizer_image_token(structured_conversation, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    return input_ids, image_tensor, image_size, structured_conversation

def give_options(input_string):
    parts = input_string.split("(")
    result = [part.split(")")[1].strip() for part in parts[1:]]
    return result

def eval_model(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Model
    # disable_torch_init()  # DO NOT ENABLE THIS: KILLS PERFORMANCE
    print(f"Using model path: {args.model_path}")
    # Load model, tokenizer, and config
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
    # get images paths
    images = {}
    file_path = hf_hub_download(repo_id="teowu/LLVisionQA-QBench", filename="images_llvisionqa.tar", repo_type="dataset")
    extract_dir = os.path.dirname(file_path)
    if not os.path.exists(extract_dir+"/images"):
        # extract
        with tarfile.open(file_path, "r:") as tar:
            # Extract all contents to the directory where the tar file is located
            tar.extractall(path=extract_dir)
    
    files_in_dir = os.listdir(extract_dir+"/images/")
    for file in files_in_dir:
        images[file] = extract_dir+"/images/"+file

    # questions
    questions = []
    dev_file_path = hf_hub_download(repo_id="teowu/LLVisionQA-QBench", filename="llvisionqa_dev.json", repo_type="dataset")
    # test_file_path = hf_hub_download(repo_id="teowu/LLVisionQA-QBench", filename="llvisionqa_test.json", repo_type="dataset")
    
    with open(dev_file_path, "r") as json_file:
        questions.extend(json.load(json_file))
    # with open(test_file_path, "r") as json_file:
    #     questions.extend(json.load(json_file))
    
    answers_file = os.path.expanduser(args.answers_file)
    if not answers_file.endswith(".jsonl"):
        raise ValueError("Answers file must be a jsonl file")

    basename = os.path.basename(answers_file)
    basename = os.path.splitext(basename)[0]
    answers_dir = os.path.dirname(answers_file)
    chunk_fname = f"{basename}_{args.chunk_idx}.jsonl"
    chunk_file = os.path.join(answers_dir, chunk_fname)
    os.makedirs(os.path.dirname(chunk_file), exist_ok=True)

    ans_file = open(chunk_file, "w")

    idx = -1
    valid_chunk = get_chunk(len(questions), args.num_chunks, args.chunk_idx)
    print(valid_chunk)
    for line in tqdm(questions, total=len(questions)):
        idx = idx+1
        if idx<valid_chunk[0] or idx>valid_chunk[1]:
            continue
        
        input_ids, image_tensor, image_sizes, prompt = process(line, args, tokenizer, image_processor, config, images)
        answer = line["correct_ans"] 
        options = line["candidates"]
        reverse_options = {}
        for ind, option in enumerate(options):
            reverse_options[option] = chr(ord('A')+ind)
        #gt_answer
        gt_answer = reverse_options[answer]
        qn_type = line["type"]
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        with torch.inference_mode():
            generated_text = generator.generate(
                [(prompt[0]["content"], image_tensor)] if image_tensor is not None else [(prompt[0]["content"], None)]
            )[0]
        if isinstance(generated_text, list):
            generated_text = generated_text[0]
        ans_file.write(json.dumps({
            "question_id": idx,
            "prompt": prompt,
            "answer":generated_text,
            "gt_answer": gt_answer,
            "type": qn_type,
            "model_id": args.model_path
        }) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/fsx-checkpoints/yashs/plm/plm_1b_cambrian7M_stage2_1024_baseline1/checkpoints/0000007000/")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--answers_file", type=str, default="./answers/answers.jsonl")
    parser.add_argument("--question_extension", type=str, default="Please answer directly with only the letter of the correct option and nothing else.")
    parser.add_argument("--conv_mode", type=str, default="llama_3")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    eval_model(args)

