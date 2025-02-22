# Diffusion-Sampler Tasks

## Overview
This repository provides support for **many_well** and **25gmm** energy functions as benchmarks for diffusion samplers. Our experiments demonstrate that an adaptive teacher can effectively serve as an off-policy trainer for diffusion-based sampling. For the results, see section 5.2 of [our paper](https://arxiv.org/abs/2410.01432).

> **Note:** The examples here are primarily for toy comparisons of off-policy behavior strategies. In real-world applications, performance heavily depends on your neural architecture and parameterization choices.

## Usage

First, navigate to the `diffusion/` directory:
```bash
cd diffusion/
```

### On-Policy Diffusion Sampler
```bash
python train.py --energy many_well
```

### Prioritized Experience Replay
```bash
python train.py --energy many_well --per
```

### Adaptive Teacher
```bash
python train.py --energy many_well --teacher
```


## Citation

If you use this code in your work, please cite our paper:

```bibtex
@article{kim2025adaptive, 
  title={Adaptive teachers for amortized samplers},
  author={Kim, Minsu and Choi, Sanghyeok and Yun, Taeyoung and Bengio, Emmanuel and Feng, Leo and Rector-Brooks, Jarrid and Ahn, Sungsoo and Park, Jinkyoo and Malkin, Nikolay and Bengio, Yoshua},
  journal={International Conference on Learning Representations (ICLR)}, 
  year={2025} 
}
