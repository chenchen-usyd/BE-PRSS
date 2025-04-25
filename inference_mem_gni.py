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

class Newpipe(StableDiffusionPipeline):
    def _encode_prompt(self,*args, **kwargs):
        embedding = super()._encode_prompt(*args,**kwargs)
        return embedding + self.noiselam * torch.randn_like(embedding)

def main(args):
    # table = None

    # if args.with_tracking:
    #     wandb.init(
    #         project="diffusion_memorization", name=args.run_name, tags=["run_mem"]
    #     )
    #     wandb.config.update(args)
    #     table = wandb.Table(
    #         columns=[
    #             "gt_prompt",
    #             "gen_prompt",
    #             # "gt_clip_score",
    #             # "gen_clip_score",
    #             "SSCD_sim",
    #             "SSCD_sim_max",
    #             "SSCD_sim_min",
    #         ]
    #     )

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

    if args.rand_noise_lam is not None: 
        
        pipe = Newpipe.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe.noiselam = args.rand_noise_lam

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    # dataset
    set_random_seed(args.gen_seed)
    dataset, prompt_key = get_dataset_finetune(args.dataset)
    if args.run_version == "ours_A" or args.run_version == "ours_B" or args.run_version == "ours_AB":
        # attn_scores = np.load(f"attn_data/memorized_prompts_baseline_attention_scores.npz")["attention_scores"]
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
        #seed = seed // 16 # for reducing batch size when loading repeated jsonl file
        torch.cuda.empty_cache()

        gt_prompt = dataset[i][prompt_key]

        ### mask
        if args.run_version == "ours_AB":
            token_len = min(len(tokenizer.encode(gt_prompt)), attn_scores.shape[-1])
            attn_score = attn_scores[i]
            attn_score_aver = np.mean(attn_score, axis=0) # size [4, 4096, 77]
            attn_token_aver = np.mean(attn_score_aver, axis=1)
            attn_score_aver = attn_score_aver.reshape(args.num_images_per_prompt, 64, 64, 77)
            attn_score_aver = attn_score_aver[:,:,:,token_len-1] # size [4, 64, 64]
            attn_score_aver = np.expand_dims(attn_score_aver, axis=1) # size [4, 1, 64, 64]
            #attn_score_aver = np.tile(attn_score_aver, (1, 4, 1, 1)) # size [4, 4, 64, 64]
            attn_score_aver = torch.from_numpy(attn_score_aver).to(device)
        elif args.run_version == "ours_A":
            token_len = min(len(tokenizer.encode(gt_prompt)), attn_scores.shape[-1])
            attn_score = attn_scores[i]
            attn_score_aver = np.mean(attn_score, axis=0) # size [4, 4096, 77]
            attn_token_aver = None
            attn_score_aver = attn_score_aver.reshape(args.num_images_per_prompt, 64, 64, 77)
            attn_score_aver = attn_score_aver[:,:,:,token_len-1] # size [4, 64, 64]
            attn_score_aver = np.expand_dims(attn_score_aver, axis=1) # size [4, 1, 64, 64]
            #attn_score_aver = np.tile(attn_score_aver, (1, 4, 1, 1)) # size [4, 4, 64, 64]
            attn_score_aver = torch.from_numpy(attn_score_aver).to(device)
        elif args.run_version == "ours_B":
            token_len = min(len(tokenizer.encode(gt_prompt)), attn_scores.shape[-1])
            attn_score = attn_scores[i]
            attn_score_aver = np.mean(attn_score, axis=0) # size [4, 4096, 77]
            attn_token_aver = np.mean(attn_score_aver, axis=1)
            attn_score_aver = None
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
                original_prompt_embeds=original_prompt_embeds, # PRSS
                m_trigger=m_trigger, # PRSS
                l_target_ss=args.l_target_ss, # PRSS
                l_target_pr=args.l_target_pr, # PRSS
                mag_thres_pr=args.mag_thres_pr, # PRSS
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
                original_prompt_embeds=original_prompt_embeds, # PRSS
                m_trigger=m_trigger, # PRSS
                l_target_ss=args.l_target_ss, # PRSS
                l_target_pr=args.l_target_pr, # PRSS
                mag_thres_pr=args.mag_thres_pr, # PRSS
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
        # all_gt_images.append(gt_images)
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
        
        # del gt_images, all_gt_images
        # import gc
        # gc.collect()
        # torch.cuda.empty_cache()

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

    # similarity model
    # sim_model = torch.jit.load("sscd_disc_large.torchscript.pt").to(device)
    # del sim_model
    # import gc
    # gc.collect()
    # torch.cuda.empty_cache()

    # reference model
    # if args.reference_model is not None:
    #     ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(
    #         args.reference_model,
    #         pretrained=args.reference_model_pretrain,
    #         device=device,
    #     )
    #     ref_tokenizer = open_clip.get_tokenizer(args.reference_model)

    # eval
    # print("eval")
    # gt_clip_scores = []
    # gen_clip_scores = []
    # SSCD_sims = []
    # SSCD_sims_max = []
    # SSCD_sims_min = []

    # for i in tqdm(range(len(all_gen_images))):
    #     gen_images = all_gen_images[i]
    #     gt_images = all_gt_images[i]
    #     prompt = all_gen_prompts[i]
    #     gt_prompt = all_gt_prompts[i]

    #     ### SSCD sim
    #     SSCD_sim = measure_SSCD_similarity(gt_images, gen_images, sim_model, device)
    #     gt_image = gt_images[SSCD_sim.argmax(dim=0)[0].item()]
    #     SSCD_sim = SSCD_sim.max(0).values
    #     SSCD_sim_max = SSCD_sim.max().item()
    #     SSCD_sim_min = SSCD_sim.min().item()
    #     SSCD_sim = SSCD_sim.mean().item()

    #     SSCD_sims.append(SSCD_sim)
    #     SSCD_sims_max.append(SSCD_sim_max)
    #     SSCD_sims_min.append(SSCD_sim_min)

    # del gen_images, gt_images, prompt, gt_prompt, SSCD_sim, gt_image, SSCD_sim_max, SSCD_sim_min, SSCD_sims, SSCD_sims_max, SSCD_sims_min
    # torch.cuda.empty_cache()

        ### clip score
        # if args.reference_model is not None:
        #     sims = measure_CLIP_similarity(
        #         [gt_image] + gen_images,
        #         gt_prompt,
        #         ref_model,
        #         ref_clip_preprocess,
        #         ref_tokenizer,
        #         device,
        #     )
        #     gt_clip_score = sims[0:1].mean().item()
        #     gen_clip_score = sims[1:].mean().item()
        # else:
        #     gt_clip_score = 0
        #     gen_clip_score = 0

        # gt_clip_scores.append(gt_clip_score)
        # gen_clip_scores.append(gen_clip_score)

        # if args.with_tracking:
        #     table.add_data(
        #         gt_prompt,
        #         prompt,
        #         # gt_clip_score,
        #         # gen_clip_score,
        #         SSCD_sim,
        #         SSCD_sim_max,
        #         SSCD_sim_min,
        #     )

    # if args.with_tracking:
    #     wandb.log({"Table": table})
    #     wandb.log(
    #         {
    #             # "gt_clip_score_mean": mean(gt_clip_scores),
    #             # "gen_clip_score_mean": mean(gen_clip_scores),
    #             "SSCD_sim_mean": mean(SSCD_sims),
    #             "SSCD_sim_max_mean": mean(SSCD_sims_max),
    #             "SSCD_sim_min_mean": mean(SSCD_sims_min),
    #         }
    #     )

    # print(f"gt_clip_score_mean: {mean(gt_clip_scores)}")
    # print(f"gen_clip_score_mean: {mean(gen_clip_scores)}")
    # print(f"SSCD_sim_mean: {mean(SSCD_sims)}")
    # print(f"SSCD_sim_max_mean: {mean(SSCD_sims_max)}")
    # print(f"SSCD_sim_min_mean: {mean(SSCD_sims_min)}")


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

    parser.add_argument("--rand_noise_lam", default=None, type=float)

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
