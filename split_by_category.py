import orjson

input_path = "/fsx-checkpoints/yashs/datasets/Cambrian-10M/jsons/Cambrian7M_shuffled.jsonl"
output_dir = "/fsx-checkpoints/yashs/datasets/Cambrian-10M/jsons/7M_by_category"
file_handles = dict()

dataset_to_category_mapping = {
    "pathvqa": "science",
    "filtered_data_engine": "science",
    "scienceqa": "science",
    "random_3rd_dvqa": "ocr",
    "synthdog_500k": "ocr",
    "st_vqa": "ocr",
    "arxivqa": "ocr",
    "clean_llava_instruct_150k_llavar": "ocr",
    "robut_wikisql": "ocr",
    "screenqa": "ocr",
    "robut_wtq": "ocr",
    "ai2d": "ocr",
    "docvqa": "ocr",
    "tat_qa": "ocr",
    "chartqa": "ocr",
    "chart2text": "ocr",
    "iconqa": "ocr",
    "rendered_text": "ocr",
    "finqa": "ocr",
    "tabmwp": "ocr",
    "infographic_vqa": "ocr",
    "vistext": "ocr",
    "hitab": "ocr",
    "raven": "math",
    "geo170k": "math",
    "geomverse": "math",
    "mathvision": "math",
    "tqa": "math",
    "intergps": "math",
    "orca": "language",
    "mathinstruct": "language",
    "code_feedback": "language",
    "orca_math": "language",
    "wizardlm": "language",
    "sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k": "general",
    "lnqa": "general",
    "lvis_instruct4v": "general",
    "qalign": "general",
    "q-instruct": "general",
    "visual7w": "general",
    "vizwiz": "general",
    "gpt77k": "general",
    "alfworldgpt": "general",
    "sketchyvqa": "general",
    "oodvqa": "general",
    "hateful_memes": "general",
    "laion_gpt4v": "general",
    "idk": "general",
    "visualmrc": "general",
    "allava-laion-500k": "general",
    "allava-vflan-200k": "general",
    "clevr": "counting",
    "tallyqa": "counting",
    "datikz": "code",
    "websight": "code",
    "filtered websight": "code",
    "design2code": "code"
}

with open(input_path, "rb") as f:
    for i, line in enumerate(f):
        row = orjson.loads(line)
        dataset_name = row["source"].split(".")[0]
        if dataset_name == "idefics375k":
            dataset_name = row["id"]
        if dataset_name in ["allava-laion-500k", "allava-vflan-200k"]:
            pass  # Skip these categories
        else:
            if "_" in dataset_name:
                dataset_name = dataset_name.split("_")[:-1]
                dataset_name = "_".join(dataset_name)
        if dataset_name == '':
            print(row)
            break
            
        category = dataset_to_category_mapping[dataset_name]
        if category not in file_handles:
            file_handles[category] = open((f"{output_dir}/{category}.jsonl"), "a")
        file_handles[category].write(orjson.dumps(row).decode("utf-8") + "\n")
        if i % 1000_000 == 0:
            print(f"Processed {i:,} rows...")