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

Please refer to each subfolder for more information.


## Installation
Use python 3.7 for compatibility in the Biochemical Discovery tasks.

To install the required dependencies, run:

~~~bash
pip install -r requirements.txt
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
