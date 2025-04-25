import argparse
import wandb
import copy
from tqdm import tqdm
from statistics import mean
from PIL import Image

import torch

import open_clip
from optim_utils import *
from io_utils import *
import mediapy as media 

from local_sd_pipeline import LocalStableDiffusionPipeline
from diffusers import DDIMScheduler, UNet2DConditionModel
from diffusers import StableDiffusionPipeline

def main(args):
    # load diffusion model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = StableDiffusionPipeline.from_pretrained(args.model_id).to(device).tokenizer

    if args.unet_id is not None:
        unet = UNet2DConditionModel.from_pretrained(
            args.unet_id, torch_dtype=torch.bfloat16
        )
        pipe = LocalStableDiffusionPipeline.from_pretrained(
            args.model_id,
            unet=unet,
            torch_dtype=torch.bfloat16,
            safety_checker=None,
            requires_safety_checker=False,
        )
    else:
        pipe = LocalStableDiffusionPipeline.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            safety_checker=None,
            requires_safety_checker=False,
        )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    # dataset
    set_random_seed(args.gen_seed)
    dataset, prompt_key = get_dataset_finetune(args.dataset)
    if args.run_version == "ours_A":
        attn_scores = np.load(f"attn_data/mit_attn_scores.npz")["attention_scores"]
        
    args.end = min(args.end, len(dataset))

    # generation
    print("generation")
    all_gen_images = []
    all_gt_images = []
    all_gen_prompts = []
    all_gt_prompts = []

    for i in tqdm(range(args.start, args.end)):
        seed = i + args.gen_seed
        torch.cuda.empty_cache()

        gt_prompt = dataset[i][prompt_key]

        ### mask
        if args.run_version == "ours_A":
            token_len = min(len(tokenizer.encode(gt_prompt)), attn_scores.shape[-1])
            attn_score = attn_scores[i]
            attn_score_aver = np.mean(attn_score, axis=0) 
            attn_token_aver = None
            attn_score_aver = attn_score_aver.reshape(args.num_images_per_prompt, 64, 64, 77)
            attn_score_aver = attn_score_aver[:,:,:,token_len-1] 
            attn_score_aver = np.expand_dims(attn_score_aver, axis=1) 
            attn_score_aver = torch.from_numpy(attn_score_aver).to(device)
        else:
            attn_score_aver = None
            attn_token_aver = None

        ### prompt modification
        if args.prompt_aug_style is not None:
            prompt = prompt_augmentation(
                gt_prompt,
                args.prompt_aug_style,
                tokenizer=pipe.tokenizer,
                repeat_num=args.repeat_num,
            )
            print(f"GT Prompt: {gt_prompt}")
            print(f"Augmented Prompt: {prompt}")
        else:
            prompt = gt_prompt

        ### optim prompt
        if args.optim_target_loss is not None:
            set_random_seed(seed)
            auged_prompt_embeds, original_prompt_embeds, m_trigger = pipe.aug_prompt(
                prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                num_images_per_prompt=args.num_images_per_prompt,
                target_steps=[args.optim_target_steps],
                lr=args.optim_lr,
                optim_iters=args.optim_iters,
                target_loss=args.optim_target_loss,
                attn_mask=attn_score_aver,
                attn_token_mask=attn_token_aver,
            )

            ### generation
            set_random_seed(seed)
            outputs = pipe(
                prompt_embeds=auged_prompt_embeds,
                original_prompt_embeds=original_prompt_embeds, 
                m_trigger=m_trigger, 
                l_target_ss=args.l_target_ss, 
                l_target_pr=args.l_target_pr, 
                mag_thres_pr=args.mag_thres_pr, 
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                num_images_per_prompt=args.num_images_per_prompt,
                attn_mask=attn_score_aver,
            ) 
        elif args.l_target_ss is not None:
            set_random_seed(seed)
            ss_prompt = [dataset[i][f"caption{j}"] for j in range(1, 26) if f"caption{j}" in dataset[i]]
            auged_prompt_embeds, original_prompt_embeds, m_trigger = pipe.aug_prompt_ss(
                prompt,
                ss_prompt=ss_prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                num_images_per_prompt=args.num_images_per_prompt,
                target_steps=[args.optim_target_steps],
                target_loss=args.l_target_ss,
                print_optim=True,
                attn_mask=attn_score_aver,
                attn_token_mask=attn_token_aver,
            )

            ### generation
            set_random_seed(seed)
            outputs = pipe(
                prompt_embeds=auged_prompt_embeds,
                original_prompt_embeds=original_prompt_embeds, 
                m_trigger=m_trigger, 
                l_target_ss=args.l_target_ss, 
                l_target_pr=args.l_target_pr, 
                mag_thres_pr=args.mag_thres_pr, 
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                num_images_per_prompt=args.num_images_per_prompt,
                attn_mask=attn_score_aver,
            )
        else:
            outputs = pipe(
                prompt=prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                num_images_per_prompt=args.num_images_per_prompt,
            )

        gen_images = outputs.images

        if "groundtruth" in args.dataset:
            gt_images = []

            curr_index = dataset[i]["index"]
            for filename in glob.glob(f"{args.dataset}/gt_images/{curr_index}/*.png"):
                im = Image.open(filename)
                gt_images.append(im)
        else:
            gt_images = [dataset[i]["image"]]
        
        all_gen_images.append(gen_images)
        all_gen_prompts.append(prompt)
        all_gt_prompts.append(gt_prompt)

        # save images as png files
        if args.store_img_png == True:
            directory_store_img_png = "images"
            if not os.path.exists(directory_store_img_png):
                os.makedirs(directory_store_img_png)

            existing_files = len([name for name in os.listdir(directory_store_img_png) if name.endswith(".png")])
            start_idx = existing_files + 1

            for idx, img in enumerate(gen_images):
                filename = f"{directory_store_img_png}/{args.run_name}_generated_image_{start_idx + idx}.png"
                media.write_image(filename, img)        

    # Save all images
    if args.store_img_npz == True:    
        directory_store_img_npz = "images_data"
        if not os.path.exists(directory_store_img_npz):
            os.makedirs(directory_store_img_npz)
        all_images_np = np.array(all_gen_images) 
        np.savez(f"{directory_store_img_npz}/{args.run_name}_generated_images.npz", images=all_images_np)


    pipe = pipe.to(torch.device("cpu"))
    del pipe
    if "pez_model" in args:
        pez_model = args.pez_model.to(torch.device("cpu"))
        del pez_model
        del args.pez_model
    torch.cuda.empty_cache()

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
    parser.add_argument("--reference_model", default=None)
    parser.add_argument("--reference_model_pretrain", default="laion2b_s12b_b42k")
    parser.add_argument("--gen_seed", default=0, type=int)
    parser.add_argument("--store_img_npz", default=True, type=bool)
    parser.add_argument("--store_img_png", default=True, type=bool)

    # mitigation strategy
    # baseline
    parser.add_argument(
        "--prompt_aug_style", default=None
    )  # rand_numb_add (RNA), rand_word_add (RT), rand_word_repeat (CWR)
    parser.add_argument("--repeat_num", default=1, type=int)

    # pe
    parser.add_argument("--optim_target_steps", default=0, type=int)
    parser.add_argument("--optim_lr", default=0.05, type=float)
    parser.add_argument("--optim_iters", default=10, type=int)
    parser.add_argument("--optim_target_loss", default=None, type=float)

    # prss
    parser.add_argument("--l_target_ss", default=None, type=float)
    parser.add_argument("--l_target_pr", default=None, type=float)
    parser.add_argument("--mag_thres_pr", default=5.0, type=float)

    args = parser.parse_args()

    main(args)
