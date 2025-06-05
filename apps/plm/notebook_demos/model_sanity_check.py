import torch
from PIL import Image
import time
# from IPython.display import HTML
# from base64 import b64encode
import textwrap

from core.args import dataclass_from_dict
from core.transforms.image_transform import get_image_transform
from core.transforms.video_transform import get_video_transform
from apps.plm.generate import PackedCausalTransformerGeneratorArgs, PackedCausalTransformerGenerator, load_consolidated_model_and_tokenizer
        

def main():
    ckpts = ["/fsx-checkpoints/yashs/dumps/naive_baselines/naive_baselines_run002/checkpoints/0000007000", # no ocr
            "/fsx-checkpoints/yashs/dumps/naive_baselines/naive_baselines_run000/checkpoints/0000007000"  # only language
    ]
    for ckpt in ckpts:
        media_paths = [
                    # "/home/yashs/projects/perception_models/apps/plm/notebook_demos/venn_diagram.png",
                    "/home/yashs/projects/perception_models/apps/plm/notebook_demos/top_scorer.png",
                    # "/home/yashs/projects/perception_models/apps/plm/notebook_demos/weather.png"
            ]
        print(f"Loading model from ckpt: {ckpt}")
        for media_path in media_paths:
            print(f"Processing media: {media_path}")
            model, tokenizer, config = load_consolidated_model_and_tokenizer(ckpt, 
                        tokenizer_path="/home/yashs/projects/perception_models/facebook/Perception-LM-1B/tokenizer.model")

            question = "Describe the image in details. Which players are at the top and bottom?"
            # display(Image.open(media_path))
            print("Generating...")
            # with basic colab we can only run with with 1 to 4 tiles, instead of full 36 tiles.
            # generate(media_path=media_path, question=question, media_type="image")
            print("Generating with 4 tiles + 1 tumb...")
            # generate(media_path=media_path, question=question, number_of_tiles=4, media_type="image")
            media_type="image"
            number_of_tiles=4
            prompts = []
            number_of_frames=4
            number_of_tiles=1
            temperature=0.0
            top_p=None
            top_k=None
            if media_type == "image":
                transform = get_image_transform(
                    vision_input_type=(
                        "vanilla" if number_of_tiles == 1 else config.data.vision_input_type
                    ),
                    image_res=model.vision_model.image_size,
                    max_num_tiles=number_of_tiles,
                )
                image = Image.open(media_path).convert("RGB")
                image, _ = transform(image)
                prompts.append((question, image))
            elif media_type == "video":
                transform = get_video_transform(
                    image_res=model.vision_model.image_size,
                )
                video_info = (media_path, number_of_frames, None, None, None)
                frames, _ = transform(video_info)
                prompts.append((question, frames))
            else:
                raise NotImplementedError(
                    f"The provided generate function only supports image and video."
                )
            # Create generator
            gen_cfg = dataclass_from_dict(
                PackedCausalTransformerGeneratorArgs,
                {"temperature": temperature, "top_p": top_p, "top_k": top_k},
                strict=False,
            )
            generator = PackedCausalTransformerGenerator(gen_cfg, model, tokenizer)
            # Run generation
            start_time = time.time()
            generation, loglikelihood, greedy = generator.generate(prompts)
            end_time = time.time()
            for _, gen in enumerate(generation):
                # Calculate tokens per second
                total_tokens = sum(
                    len(tokenizer.encode(gen, False, False)) for gen in generation
                )
                tokens_per_second = total_tokens / (end_time - start_time)
                print("=================================================")
                print(textwrap.fill(gen, width=75))
                print(f"Tokens per second: {tokens_per_second:.2f}")
                print("=================================================")
            print("Done generating.")
        print("=================================================")
        print("\n \n\n")
    


if __name__ == "__main__":
    main()