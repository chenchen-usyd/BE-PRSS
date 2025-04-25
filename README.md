# Memorization in Image Diffusion Models
Official repo for [ICLR 2025 (Spotlight): Exploring Local Memorization in Diffusion Models via Bright Ending Attention](https://openreview.net/forum?id=p4cLtzk4oe) and [CVPR 2025: Enhancing Privacy-Utility Trade-offs to Mitigate Memorization in Diffusion Models](https://cvpr.thecvf.com/virtual/2025/poster/34842).

Authors: Chen Chen, Daochang Liu, Mubarak Shah, Chang Xu

If you have any questions or suggestions, please reach out to Chen (<cche0711@uni.sydney.edu.au>).

## Dependencies
- PyTorch == 1.13.0
- transformers == 4.30.2
- diffusers == 0.18.2
- accelerate == 0.21.0
- datasets


## Extract the BE memorization mask
To obtain the BE attention scores, run the following command to store the results in the `attn_data` directory:

```
python detect_mem.py --run_name memorized_prompts --run_version baseline --dataset examples/sdv1_500_memorized.jsonl --gen_seed 0 --model_id PATH_TO_SD_MODEL --num_images_per_prompt 16
```

## Detect memorization
To obtain the memorization detection results, run the following command to store the results in a new `det_outputs` directory and the generated images in the `images_data` directory:

```python
python detect_mem.py --run_name memorized_prompts --run_version ours --dataset examples/sdv1_500_memorized.jsonl --gen_seed 0 --model_id PATH_TO_SD_MODEL --num_images_per_prompt 16
```

## Mitigate memorization
To obtain the memorization mitigation results, run the following command to store the generated images after using the mitigation strategy in the `images_data` directory:

### Baseline ([Wen et al., ICLR 2024](https://openreview.net/forum?id=84n3UwkH7b))
```
python inference_mem.py \
    --run_name baseline \
    --run_version baseline \
    --dataset sdv1_500_mem_groundtruth \
    --gen_seed 0 \
    --model_id PATH_TO_SD_MODEL \
    --reference_model ViT-g-14 \
    --with_tracking \
    --num_images_per_prompt 1 \
    --optim_target_steps 0 \
    --optim_target_loss 1
```

### Baseline + BE ([Chen et al., ICLR 2025](https://openreview.net/forum?id=p4cLtzk4oe))
```
python inference_mem.py \
    --run_name ours_A \
    --run_version ours_A \
    --dataset sdv1_500_mem_groundtruth \
    --gen_seed 0 \
    --model_id PATH_TO_SD_MODEL \
    --reference_model ViT-g-14 \
    --with_tracking \
    --num_images_per_prompt 1 \
    --optim_target_steps 0 \
    --optim_target_loss 1
```

### PRSS ([Chen et al., CVPR 2025](https://cvpr.thecvf.com/virtual/2025/poster/34842))
```
python inference_mem.py \
    --run_name baseline \
    --run_version baseline \
    --dataset sdv1_500_mem_groundtruth \
    --gen_seed 0 \
    --model_id PATH_TO_SD_MODEL \
    --reference_model ViT-g-14 \
    --with_tracking \
    --num_images_per_prompt 1 \
    --optim_target_steps 0 \
    --l_target_pr 3 \
    --mag_thres_pr 5 \
    --l_target_ss 1
```

### PRSS + BE ([Chen et al., CVPR 2025](https://cvpr.thecvf.com/virtual/2025/poster/34842))
```
python inference_mem.py \
    --run_name ours_A \
    --run_version ours_A \
    --dataset sdv1_500_mem_groundtruth \
    --gen_seed 0 \
    --model_id PATH_TO_SD_MODEL \
    --reference_model ViT-g-14 \
    --with_tracking \
    --num_images_per_prompt 1 \
    --optim_target_steps 0 \
    --l_target_pr 3 \
    --mag_thres_pr 5 \
    --l_target_ss 1
```



## To-do
Additional example notebooks will be provided shortly to make it easy to get started and explore outputs with minimal setup.


## Acknowledgements

Our implementation builds on the source code from the following repositories:

* [Detecting, Explaining, and Mitigating Memorization in Diffusion Models](https://github.com/YuxinWenRick/diffusion_memorization)

* [Understanding and Mitigating Copying in Diffusion Models](https://github.com/somepago/DCR/tree/main)

* [Prompt-to-Prompt Image Editing with Cross-Attention Control](https://github.com/google/prompt-to-prompt/)

Thanks for their awesome works!


## Cite our work
If you find ours works useful, please cite our paper:

```bibtex
@inproceedings{chen2025exploring,
  title={Exploring Local Memorization in Diffusion Models via Bright Ending Attention},
  author={Chen, Chen and Liu, Daochang and Shah, Mubarak and Xu, Chang},
  booktitle={ICLR},
  year={2025}
}

@inproceedings{chen2025enhancing,
  title={Enhancing Privacy-Utility Trade-offs to Mitigate Memorization in Diffusion Models},
  author={Chen, Chen and Liu, Daochang and Shah, Mubarak and Xu, Chang},
  booktitle={CVPR},
  year={2025}
}
```