# Adaptive Teachers for Amortized Samplers

**Authors:**  
Minsu Kim, Sanghyeok Choi, Taeyoung Yun, Emmanuel Bengio, Leo Feng, Jarrid Rector-Brooks, Sungsoo Ahn, Jinkyoo Park, Nikolay Malkin, Yoshua Bengio  

**Paper Link:** [https://arxiv.org/abs/2410.01432](https://arxiv.org/abs/2410.01432)

## Introduction
This repository contains the official code for the paper:  
**“Adaptive Teachers for Amortized Samplers”** (ICLR 2025).

Amortized inference aims to train a parametric model to approximate a target distribution where direct sampling is typically intractable. We view sampling as a sequential decision-making process and leverage reinforcement learning (RL) techniques such as Generative Flow Networks. However, efficient exploration remains challenging.  
Our key innovation is an auxiliary "Teacher" behavior model that adapts its sampling distribution to focus on regions where the "Student" amortized sampler has high error. This Teacher distribution can generalize across unexplored modes, improving both sample efficiency and mode coverage.

This repository is organized as follows:
- **`grid/`**: Benchmarking on a deceptive gridworld environment with multiple reward modes and deceptive regions.  
- **`diffusion/`**: Implementation for diffusion-based sampling tasks.  
- **`discovery/`**: Biochemical discovery tasks demonstrating the method's ability to discover diverse, high-reward candidates.

---

## Installation
To install the required dependencies, run:

~~~bash
pip install --upgrade ...
~~~
*(Please fill in the appropriate dependency names and versions.)*

---

## Grid Environment

### Overview
The `grid/` folder contains code to run experiments on the deceptive gridworld environment where the agent needs to explore effectively to discover multiple high-reward modes.

### Usage

~~~bash
python train_grid.py --some_flag ...
~~~
*(Please fill in the appropriate script name and command-line flags.)*

---


# Diffusion-Sampler Tasks

## Overview
This repository provides support for **many_well** and **25gmm** energy functions as benchmarks for diffusion samplers. Our experiments demonstrate that an adaptive teacher can effectively serve as an off-policy trainer for diffusion-based sampling.

> **Note:** The examples here are primarily for toy comparisons of off-policy behavior strategies. In real-world applications, performance heavily depends on your neural architecture and parameterization choices.

## Usage

### On-Policy Diffusion Sampler
```bash
cd diffusion/
python train.py --energy many_well
```

### Prioritized Experience Replay
```bash
cd diffusion/
python train.py --energy many_well --per
```

### Adaptive Teacher
```bash
cd diffusion/
python train.py --energy many_well --teacher
```



---

## Biochemical Discovery Tasks

### Overview
The `discovery/` folder provides code for four biochemical discovery tasks, showcasing how our adaptive Teacher-Student approach can improve mode coverage and sample efficiency in real-world discovery scenarios.

### Usage

~~~bash
python run_discovery.py --data_path ... --model_config ...
~~~
*(Please fill in the appropriate script name and command-line flags.)*

---

## Citation

If you use this code in your work, please cite our paper:

```bibtex
@article{kim2025adaptive, 
  title={Adaptive teachers for amortized samplers},
  author={Kim, Minsu and Choi, Sanghyeok and Yun, Taeyoung and Bengio, Emmanuel and Feng, Leo and Rector-Brooks, Jarrid and Ahn, Sungsoo and Park, Jinkyoo and Malkin, Nikolay and Bengio, Yoshua},
  journal={International Conference on Learning Representations (ICLR)}, 
  year={2025} 
}
