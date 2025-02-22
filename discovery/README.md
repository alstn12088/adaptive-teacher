# Biochemical Discovery Tasks

## Overview
The `discovery/` folder provides code for four biochemical discovery tasks, showcasing how our adaptive Teacher-Student approach can improve mode coverage and sample efficiency in real-world discovery scenarios. For the results, see section 5.3 of [our paper](https://arxiv.org/abs/2410.01432).

## Setup
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


## Citation

If you use this code in your work, please cite our paper:

```bibtex
@article{kim2025adaptive, 
  title={Adaptive teachers for amortized samplers},
  author={Kim, Minsu and Choi, Sanghyeok and Yun, Taeyoung and Bengio, Emmanuel and Feng, Leo and Rector-Brooks, Jarrid and Ahn, Sungsoo and Park, Jinkyoo and Malkin, Nikolay and Bengio, Yoshua},
  journal={International Conference on Learning Representations (ICLR)}, 
  year={2025} 
}
