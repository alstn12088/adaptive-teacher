# Adaptive Teachers for Amortized Samplers

**Authors:**  
Minsu Kim*, Sanghyeok Choi*, Taeyoung Yun, Emmanuel Bengio, Leo Feng, Jarrid Rector-Brooks, Sungsoo Ahn, Jinkyoo Park, Nikolay Malkin, Yoshua Bengio  

**Paper Link:** [https://arxiv.org/abs/2410.01432](https://arxiv.org/abs/2410.01432)

## Introduction
This repository contains the official code for the paper, "**Adaptive Teachers for Amortized Samplers** (ICLR 2025)".

Amortized inference aims to train a parametric model to approximate a target distribution where direct sampling is typically intractable. We view sampling as a sequential decision-making process and leverage reinforcement learning (RL) techniques, specifically Generative Flow Networks (GFlowNets). However, efficient exploration remains challenging.  
Our key innovation is an auxiliary "Teacher" behavior model that adapts its sampling distribution to focus on regions where the "Student" amortized sampler has high error. This Teacher distribution can generalize across unexplored modes, improving both sample efficiency and mode coverage.

This repository is organized as follows:
- **`grid/`**: Benchmarking on a deceptive gridworld environment with multiple reward modes and deceptive regions.  
- **`diffusion/`**: Implementation for diffusion-based sampling tasks.  
- **`discovery/`**: Biochemical discovery tasks demonstrating the method's ability to discover diverse, high-reward candidates.


## Installation
Use python 3.7 for compatibility in the Biochemical Discovery tasks.

To install the required dependencies, run:

~~~bash
pip install -r requirements.txt
~~~



# Grid Environment

## Overview
The `grid/` folder contains code to run experiments on the deceptive gridworld environment where the agent needs to explore effectively to discover multiple high-reward modes.

## Usage

First, navigate to the `grid/` directory:
```bash
cd grid/
```

***Check `run.sh` for the full list of commands used in the paper.***

Basic usage:
```bash
python trainer.py --agent $AGENT --ndim $NDIM --horizon $HORIZON --run_name $RUN_NAME --seed $SEED
```
where,
- `$AGENT` $\in$ {"tb", "gafn", "teacher"}
- `$NDIM` $\in \mathbb{N}$
- `$HORIZON` $\in \mathbb{N}$.

To log the results and plots with wandb, add `--logger wandb --plot` to the command.  


To run with the buffers:
```bash
python trainer.py --agent $AGENT --ndim $NDIM --horizon $HORIZON --use_buffer --buffer_size $BUFSIZE --buffer_pri $BUFPRI --run_name $RUN_NAME --seed $SEED
```
where,
- `$BUFSIZE` $\in \mathbb{N}$
- `$BUFPRI` $\in$ {"none', "reward", "loss", "teacher_reward"}.


To run with the back-and-forth local search:
```bash
cd grid/
python trainer.py --agent $AGENT --ndim $NDIM --horizon $HORIZON --ls --run_name $RUN_NAME --seed $SEED
```


To run with the detailed balance (DB) loss:
```bash
python trainer_db.py --agent $AGENT --ndim $NDIM --horizon $HORIZON --run_name $RUN_NAME --seed $SEED
```
where `$AGENT` $\in$ {"db", "teacher_db"}.



# Diffusion-Sampler Tasks

## Overview
This repository provides support for **many_well** and **25gmm** energy functions as benchmarks for diffusion samplers. Our experiments demonstrate that an adaptive teacher can effectively serve as an off-policy trainer for diffusion-based sampling.

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



# Biochemical Discovery Tasks

## Overview
The `discovery/` folder provides code for four biochemical discovery tasks, showcasing how our adaptive Teacher-Student approach can improve mode coverage and sample efficiency in real-world discovery scenarios.

## Installation
### Large Files
To run `sehstr` task, you should download `sehstr_gbtr_allpreds.pkl.gz` and `block_18_stop6.pkl.gz`. Both are available for download at https://figshare.com/articles/dataset/sEH_dataset_for_GFlowNet_/22806671
DOI: 10.6084/m9.figshare.22806671
These files should be placed in `datasets/sehstr/`.

### Additional Dependencies
Biochemical discovery tasks require `torch_geometric` library. 
```
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
```

## Usage

First, navigate to the `discovery/` directory:
```bash
cd discovery/
```

Run the following commands:
~~~bash
python runexpwb.py --setting qm9str --model teacher

# With PRT
python runexpwb.py --setting qm9str --model teacher --offline_select prt

# With PER
python runexpwb.py --setting qm9str --model teacher --per True
~~~


# Citation

If you use this code in your work, please cite our paper:

```bibtex
@article{kim2025adaptive, 
  title={Adaptive teachers for amortized samplers},
  author={Kim, Minsu and Choi, Sanghyeok and Yun, Taeyoung and Bengio, Emmanuel and Feng, Leo and Rector-Brooks, Jarrid and Ahn, Sungsoo and Park, Jinkyoo and Malkin, Nikolay and Bengio, Yoshua},
  journal={International Conference on Learning Representations (ICLR)}, 
  year={2025} 
}
