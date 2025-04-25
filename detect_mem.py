import argparse
from tqdm import tqdm

import torch
import numpy 
import numpy as np
import abc 

from optim_utils import *
from io_utils import *
from register_attention_control import register_attention_control
from controller import * 
import mediapy as media 
import matplotlib.pyplot as plt 
import os 

from local_sd_pipeline import LocalStableDiffusionPipeline
from diffusers import DDIMScheduler, UNet2DConditionModel

def main(args):
    # load diffusion model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    tokenizer = StableDiffusionPipeline.from_pretrained(args.model_id).to(device).tokenizer
    attention_controller = AttentionStore()

    if args.unet_id is not None:
        unet = UNet2DConditionModel.from_pretrained(
            args.unet_id, torch_dtype=torch.float16
        )
        pipe = LocalStableDiffusionPipeline.from_pretrained(
            args.model_id,
            unet=unet,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
        )
    else:
        pipe = LocalStableDiffusionPipeline.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
        )

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    # dataset
    set_random_seed(args.gen_seed)
    dataset, prompt_key = get_dataset(args.dataset, pipe)
    if args.run_version == "ours":
        attn_scores = np.load(f"attn_data/{args.run_name}_baseline_attention_scores.npz")["attention_scores"]

    args.end = min(args.end, len(dataset))

    # generation
    print("generation")

    all_metrics = ["uncond_noise_norm", "text_noise_norm", "text_noise_norm_masked"]
    all_tracks = []
    all_images = []
    attention_scores = []

    for i in tqdm(range(args.start, args.end)):
        seed = i + args.gen_seed

        # Reset the attention controller before each prompt
        attention_controller.reset()
        
        prompt = dataset[i][prompt_key]
        
        if args.run_version == "ours":
            token_len = min(len(tokenizer.encode(prompt)), attn_scores.shape[-1])
            attn_score = attn_scores[i]
            attn_score_aver = np.mean(attn_score, axis=0) 
            attn_score_aver = attn_score_aver.reshape(args.num_images_per_prompt, 64, 64, 77)
            attn_score_aver = attn_score_aver[:,:,:,token_len-1] 
            attn_score_aver = np.expand_dims(attn_score_aver, axis=1) 
            attn_score_aver = torch.from_numpy(attn_score_aver).to(device)
        else:
            attn_score_aver = None

        ### generation
        set_random_seed(seed)
        outputs, track_stats = pipe(
            prompt,
            controller=attention_controller,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            num_images_per_prompt=args.num_images_per_prompt,
            track_noise_norm=True,
            attn_mask=attn_score_aver,
        )

        uncond_noise_norm, text_noise_norm, text_noise_norm_masked = (
            track_stats["uncond_noise_norm"],
            track_stats["text_noise_norm"],
            track_stats["text_noise_norm_masked"],
        )

        curr_line = {}
        for metric_i in all_metrics:
            values = locals()[metric_i]
            curr_line[f"{metric_i}"] = values

        curr_line["prompt"] = prompt

        all_tracks.append(curr_line)
        all_images.extend(outputs.images)

        # Collect attention maps and append to the main list
        if args.run_version == "baseline":
            attention_score = [
                attn.cpu().numpy()
                for attn in attention_controller.attention_store["down_cross"]
            ]
            attention_scores.append(attention_score)     

        # save images as png files
        if args.store_img_png == True:
            directory_store_img_png = "images"
            if not os.path.exists(directory_store_img_png):
                os.makedirs(directory_store_img_png)

            existing_files = len([name for name in os.listdir(directory_store_img_png) if name.endswith(".png")])
            start_idx = existing_files + 1

            for idx, img in enumerate(outputs.images):
                filename = f"{directory_store_img_png}/{args.run_name}_{args.run_version}_generated_image_{start_idx + idx}.png"
                media.write_image(filename, img)

    # Save all attention scores
    if args.run_version == "baseline":
        directory_store_attn_npz = "attn_data"
        if not os.path.exists(directory_store_attn_npz):
            os.makedirs(directory_store_attn_npz)
        attention_scores = np.array(attention_scores)
        np.savez(f"{directory_store_attn_npz}/{args.run_name}_{args.run_version}_attention_scores.npz", attention_scores=attention_scores)

    # Save all images
    if args.store_img_npz == True:    
        directory_store_img_npz = "images_data"
        if not os.path.exists(directory_store_img_npz):
            os.makedirs(directory_store_img_npz)
        all_images_np = np.array(all_images) 
        np.savez(f"{directory_store_img_npz}/{args.run_name}_{args.run_version}_generated_images.npz", images=all_images_np)

    os.makedirs("det_outputs", exist_ok=True)
    write_jsonlines(all_tracks, f"det_outputs/{args.run_name}_{args.run_version}.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="diffusion memorization")
    parser.add_argument("--run_name", default="test")
    parser.add_argument("--run_version", default="baseline")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=500, type=int)
    parser.add_argument("--image_length", default=512, type=int)
    parser.add_argument("--model_id", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--unet_id", default=None)
    parser.add_argument("--with_tracking", action="store_true")
    parser.add_argument("--num_images_per_prompt", default=4, type=int)
    parser.add_argument("--guidance_scale", default=7.5, type=float)
    parser.add_argument("--num_inference_steps", default=50, type=int)
    parser.add_argument("--gen_seed", default=0, type=int)
    parser.add_argument("--store_attn_npz", default=True, type=bool)
    parser.add_argument("--store_img_npz", default=True, type=bool)
    parser.add_argument("--store_img_png", default=True, type=bool)

    args = parser.parse_args()

    main(args)
