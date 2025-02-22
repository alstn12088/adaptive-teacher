ndim=$1
horizon=$2
seed=$3

# Main
python trainer.py --seed $seed --agent tb --ndim $ndim --horizon $horizon --logger wandb --plot &
python trainer.py --seed $seed --agent tb --eps 0.01 --run_name "eps0.01" --ndim $ndim --horizon $horizon --logger wandb --plot &
python trainer.py --seed $seed --agent gafntb --ri_scale 0.01 --run_name "riscale0.01" --ndim $ndim --horizon $horizon --logger wandb --plot &
python trainer.py --seed $seed --agent teacher --ndim $ndim --horizon $horizon --logger wandb --plot &
python trainer.py --seed $seed --agent tb --use_buffer --buffer_pri reward --run_name "PRT" --ndim $ndim --horizon $horizon --logger wandb --plot &
python trainer.py --seed $seed --agent tb --use_buffer --buffer_pri teacher_reward --run_name "PER" --ndim $ndim --horizon $horizon --logger wandb --plot &
python trainer.py --seed $seed --agent teacher --use_buffer --buffer_pri teacher_reward --run_name "PER" --ndim $ndim --horizon $horizon --logger wandb --plot &

wait

# Local search experiments
python trainer.py --seed $seed --agent tb --ls --run_name "ls" --ndim $ndim --horizon $horizon --logger wandb --plot &
python trainer.py --seed $seed --agent teacher --ls --run_name "ls" --ndim $ndim --horizon $horizon --logger wandb --plot &
python trainer.py --seed $seed --agent tb --ls --use_buffer --buffer_pri teacher_reward --run_name "ls_PER" --ndim $ndim --horizon $horizon --logger wandb --plot &
python trainer.py --seed $seed --agent teacher --ls --use_buffer --buffer_pri teacher_reward --run_name "ls_PER" --ndim $ndim --horizon $horizon --logger wandb --plot &

wait

# Detailed balance experimentss
python trainer_db.py --seed $seed --agent db --ndim $ndim --horizon $horizon --logger wandb --plot &
python trainer_db.py --seed $seed --agent db --use_buffer --buffer_pri teacher_reward --run_name "PER" --ndim $ndim --horizon $horizon --logger wandb --plot &
python trainer_db.py --seed $seed --agent teacher_db --ndim $ndim --horizon $horizon --logger wandb --plot &
python trainer_db.py --seed $seed --agent teacher_db --use_buffer --buffer_pri teacher_reward --run_name "PER" --ndim $ndim --horizon $horizon --logger wandb --plot &

wait
