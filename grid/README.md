# Deceptive GridWorld Tasks

## Overview
The `grid/` folder contains code to run experiments on the deceptive gridworld environment where the agent needs to explore effectively to discover multiple high-reward modes. For the results, see section 5.1 of [our paper](https://arxiv.org/abs/2410.01432).

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
- `$NDIM` $\in$ {2, 4, ...}
- `$HORIZON` $\in$ {16, 32, ...}.

To log the results and plots with wandb, add `--logger wandb --plot` to the command.  


To run with the buffers:
```bash
python trainer.py --agent $AGENT --ndim $NDIM --horizon $HORIZON --use_buffer --buffer_size $BUFSIZE --buffer_pri $BUFPRI --run_name $RUN_NAME --seed $SEED
```
where,
- `$BUFSIZE` is a natural number or -1 to use the default setting.
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


## Citation

If you use this code in your work, please cite our paper:

```bibtex
@article{kim2025adaptive, 
  title={Adaptive teachers for amortized samplers},
  author={Kim, Minsu and Choi, Sanghyeok and Yun, Taeyoung and Bengio, Emmanuel and Feng, Leo and Rector-Brooks, Jarrid and Ahn, Sungsoo and Park, Jinkyoo and Malkin, Nikolay and Bengio, Yoshua},
  journal={International Conference on Learning Representations (ICLR)}, 
  year={2025} 
}
