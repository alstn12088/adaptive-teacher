"""
Trainer for the GridWorld environment
"""
import gc
import os
import sys

sys.path.append("..")

from grid.local_search import back_and_forth

from typing import Callable, Dict, Optional
import itertools

from tqdm import tqdm
import numpy as np
import torch

torch.set_num_threads(1)
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim import Adam

from grid.buffer import TerminalStateBuffer
from grid.env import DeceptiveGridWorld
from grid.models import BaseAgent, TBAgent, GAFNTBAgent, TeacherStudentTBAgent
from grid.models.teacher import get_teacher_reward
from grid.utils import compute_empirical_distribution_error, plot_empirical_distribution, plot_target_distribution, seed_everything


class Trainer:
    def __init__(
        self,
        env: DeceptiveGridWorld,
        agent: BaseAgent,
        lr_dict: Dict[str, float],
        clip_grad_norm: float = 1.0,
        n_train_steps: int = 20000,
        batch_size: int = 16,
        ts_ratio: float = 1,
        use_buffer: bool = True,
        buffer_params: Dict = {},
        ls: bool = False,
        ls_back_ratio: float = 0.5,
        log: bool = False,
        C: float = 19.0,
        alpha: float = 0.0,
        no_tqdm: bool = False,
        num_empirical_loss: int = 100000,
        n_logs: int = 5,
        logging_fn: Optional[Callable] = None,  # wandb
        run_name: str = "",
        eval_num_empirical_loss: int = 100000,
        eval_batch_size: int = 1000,
        plot: bool = False,
    ) -> None:
        self.env = env
        self.env.get_true_density()  # precompute true density
        assert self.env.goals is not None
        self.goal_found_map = {goal: 0 for goal in self.env.goals}
        self.agent = agent

        self.buffer = TerminalStateBuffer(**buffer_params) if use_buffer else None

        # Optimizer
        self.params_list = [{"params": self.agent.model.parameters(), "lr": lr_dict["model"]}]  # type: ignore

        if isinstance(self.agent, TBAgent):  # Z for TB requires higher lr
            self.params_list.append({"params": [self.agent.log_Z], "lr": lr_dict["tb_z"]})
        elif isinstance(self.agent, GAFNTBAgent):
            self.params_list.append({"params": [self.agent.log_Z], "lr": lr_dict["tb_z"]})
            self.rnd_params_list = [
                {"params": self.agent.ri_model.parameters(), "lr": lr_dict["rnd"]},
            ]
        elif isinstance(self.agent, TeacherStudentTBAgent):
            self.params_list_teacher = [{"params": self.agent.teacher_model.parameters(), "lr": lr_dict["model"]}]
            self.params_list.extend([{"params": [self.agent.log_Z], "lr": lr_dict["tb_z"]}])
            self.params_list_teacher.extend([{"params": [self.agent.log_Z_teacher], "lr": lr_dict["tb_z"]}])
        else:
            raise NotImplementedError

        if isinstance(self.agent, GAFNTBAgent):
            self.optimizer = Adam(self.params_list + self.rnd_params_list)
        elif isinstance(self.agent, TeacherStudentTBAgent):
            self.optimizer = Adam(self.params_list)
            self.optimizer_teacher = Adam(self.params_list_teacher)
        else:
            self.optimizer = Adam(self.params_list)

        self.clip_grad_norm: float = clip_grad_norm

        # training
        self.n_train_steps: int = n_train_steps
        self.batch_size: int = batch_size
        self.ts_ratio = max(int(ts_ratio), 1)  # train / sample ratio

        # Local Search
        self.ls = ls
        self.ls_iter = 4
        self.ls_back_ratio = ls_back_ratio

        # Teacher
        self.log = log
        self.C = C
        self.alpha = alpha

        # logging
        self.no_tqdm: bool = no_tqdm
        self.num_empirical_loss: int = num_empirical_loss
        self.logging_fn = logging_fn
        self.run_name = run_name
        self.eval_interval = self.n_train_steps // n_logs
        self.eval_num_empirical_loss = eval_num_empirical_loss
        self.eval_batch_size = eval_batch_size
        self.plot = plot

    def train(self) -> None:
        # Plot the target distribution
        _, true_density, _, _ = self.env.get_true_density()
        VMAX = max(0.0, true_density.max())
        if self.plot:
            plot_target_distribution(self.env, vmax=VMAX, logging_fn=self.logging_fn)

        losses = []
        teacher_losses = []
        infos = []
        all_visited = []
        n_reward_calls = n_grad_steps = 0

        for i in tqdm(range(self.n_train_steps), disable=self.no_tqdm, dynamic_ncols=True, desc="Main Training Loop"):
            train_batch = []

            ### Teacher ###
            # Maybe use off-policy sampling, local search, or replay buffer
            if isinstance(self.agent, TeacherStudentTBAgent):
                if i % 2 == 0 or self.buffer is None:
                    if (i // 2) % 2 == 0:  # onpolicy (student)
                        _batch, _visited = self.agent.sample_many(self.batch_size)
                        n_reward_calls += self.batch_size
                    else:  # offpolicy (teacher)
                        _batch, _visited = self.agent.sample_many(self.batch_size, is_teacher=True)
                        n_reward_calls += self.batch_size
                        if self.ls and (i // 4) % 4 == 3:  # do local search
                            _batch, _visited = back_and_forth(
                                self.agent,
                                _batch,
                                self.ls_back_ratio,
                                self.ls_iter,
                                eps_noisy=False,
                                teacher_ls=True,
                                log=self.log,
                                C=self.C,
                                alpha=self.alpha,
                            )
                            n_reward_calls += self.batch_size * self.ls_iter

                else:  # Buffer
                    xs_and_rs = self.buffer.sample(self.batch_size)
                    _batch = []
                    for _x, _r in xs_and_rs:
                        _obs, _s, _a = tuple(map(torch.from_numpy, self.env.generate_backward(np.array(_x))))
                        _batch.append((_obs, _s, _a, _r))
                    _visited = []

            ### TB ###
            elif isinstance(self.agent, TBAgent):  # Maybe use local search or replay buffer
                if i % 2 == 0 or self.buffer is None:  # onpolicy
                    _batch, _visited = self.agent.sample_many(self.batch_size)
                    n_reward_calls += self.batch_size
                    if self.ls and (i // 2) % 8 == 7:  # local search
                        _batch, _visited = back_and_forth(
                            self.agent, _batch, self.ls_back_ratio, self.ls_iter, eps_noisy=False
                        )
                        n_reward_calls += self.batch_size * self.ls_iter
                else:  # Buffer
                    xs_and_rs = self.buffer.sample(self.batch_size)
                    _batch = []
                    for _x, _r in xs_and_rs:
                        _obs, _s, _a = tuple(map(torch.from_numpy, self.env.generate_backward(np.array(_x))))
                        _batch.append((_obs, _s, _a, _r))
                    _visited = []

            ### GAFN ###
            elif isinstance(self.agent, GAFNTBAgent):  # Don't use local search or replay buffer
                _batch, _visited = self.agent.sample_many(self.batch_size)
                n_reward_calls += self.batch_size

            else:
                raise NotImplementedError

            train_batch += _batch
            all_visited += _visited
            for s in _visited:
                if s in self.goal_found_map:
                    self.goal_found_map[s] += 1

            for _ in range(self.ts_ratio):
                if isinstance(self.agent, TeacherStudentTBAgent):
                    loss, discrepancies, info = self.agent.learn_from(train_batch, is_teacher=False)
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.clip_grad_norm > 0:
                        clip_grad_norm_(
                            itertools.chain(*[_l["params"] for _l in self.params_list]), self.clip_grad_norm
                        )
                    self.optimizer.step()

                    teacher_rewards = get_teacher_reward(discrepancies, train_batch, self.log, self.C, self.alpha)

                    teacher_loss, _, teacher_info = self.agent.learn_from(train_batch, is_teacher=True, new_rewards=teacher_rewards)  # type: ignore
                    self.optimizer_teacher.zero_grad()
                    teacher_loss.backward()
                    if self.clip_grad_norm > 0:
                        clip_grad_norm_(
                            itertools.chain(*[_l["params"] for _l in self.params_list_teacher]), self.clip_grad_norm
                        )
                    self.optimizer_teacher.step()

                    info.update(teacher_info)
                    teacher_losses.append(teacher_loss.item())

                elif isinstance(self.agent, TBAgent):
                    loss, discrepancies, info = self.agent.learn_from(train_batch)
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.clip_grad_norm > 0:
                        clip_grad_norm_(
                            itertools.chain(*[_l["params"] for _l in self.params_list]), self.clip_grad_norm
                        )
                    self.optimizer.step()

                    if self.buffer is not None and self.buffer.prioritized == "teacher_reward":
                        teacher_rewards = get_teacher_reward(discrepancies, train_batch, self.log, self.C, self.alpha)

                elif isinstance(self.agent, GAFNTBAgent):
                    loss, info = self.agent.learn_from(train_batch)

                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.clip_grad_norm > 0:
                        clip_grad_norm_(
                            itertools.chain(*[_l["params"] for _l in self.params_list]), self.clip_grad_norm
                        )
                    self.optimizer.step()

                else:
                    raise NotImplementedError

                n_grad_steps += 1

                losses.append(loss.item())
                if len(info) > 0:
                    infos.append(info)

            # Add to buffer
            if self.buffer is not None and len(_visited) > 0:
                rewards = [tup[3] for tup in train_batch]
                if self.buffer.prioritized == "reward":
                    priorities = [tup[3] for tup in train_batch]
                elif self.buffer.prioritized == "loss":
                    priorities = [d**2 for d in discrepancies]
                elif self.buffer.prioritized == "teacher_reward":
                    if isinstance(teacher_rewards[0], torch.Tensor):
                        priorities = [t_r.sum() for t_r in teacher_rewards]  # type: ignore
                    else:
                        priorities = teacher_rewards
                else:
                    priorities = None
                self.buffer.add_batch(_visited, rewards, priorities)  # type: ignore

            # Logging
            if i == 0 or (i + 1) % self.eval_interval == 0 or (i + 1) == self.n_train_steps:
                l1, w_l1 = compute_empirical_distribution_error(self.env, all_visited[-self.num_empirical_loss:])

                print(f"Step: {i}, Loss: {np.mean(losses[-100:]):.6f}")
                print(f"empirical L1 dist: {l1:.6f}, weighted L1 dist: {w_l1:.6f}, #visited: {len(all_visited)}")

                n_goals_found = sum([1 for v in self.goal_found_map.values() if v > 0])
                print(f"n_goals_found: {n_goals_found}/{len(self.goal_found_map)}")

                if self.logging_fn is not None:
                    self.logging_fn(
                        {
                            "loss": np.mean(losses[-100:]),
                            "empirical_l1": l1,
                            "empirical_w_l1": w_l1,
                            "n_goals_found": n_goals_found,
                            "n_reward_calls": n_reward_calls,
                            "n_grad_steps": n_grad_steps,
                        },
                        step=i,
                    )

                # Sampling and Evaluation
                self.agent.eval()

                eval_n_sampled = 0
                eval_visited, _batch = [], []
                pbar = tqdm(total=self.eval_num_empirical_loss, disable=self.no_tqdm, dynamic_ncols=True, desc="Sampling for eval")
                while eval_n_sampled < self.eval_num_empirical_loss:
                    bs = min(self.eval_batch_size, self.eval_num_empirical_loss - eval_n_sampled)
                    _batch, _eval_visited = self.agent.sample_many(bs, eval=True)
                    eval_visited += _eval_visited
                    eval_n_sampled += bs
                    pbar.update(bs)

                assert len(eval_visited) == self.eval_num_empirical_loss
                eval_l1, eval_w_l1 = compute_empirical_distribution_error(self.env, eval_visited)
                print(
                    f"Eval: empirical L1 dist: {eval_l1:.6f}, weighted L1 dist: {eval_w_l1:.6f}, #visited: {self.eval_num_empirical_loss}"
                )
                if self.logging_fn is not None:
                    self.logging_fn({"eval_empirical_l1": eval_l1, "eval_empirical_w_l1": eval_w_l1}, step=i)

                # Plot empirical distribution
                if self.plot:
                    plot_empirical_distribution(
                        self.env, eval_visited, label=f"empirical_density", logging_fn=self.logging_fn, step=i, vmax=VMAX
                    )

                del eval_visited[:], _batch[:]
                gc.collect()

                self.agent.train()

        # Save the model
        os.makedirs("pretrained", exist_ok=True)
        torch.save(self.agent.state_dict(), f"pretrained/model_d{self.env.ndim}_H{self.env.horizon}_{self.run_name}.pt")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--device_num", type=int, default=0)

    # Env
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--ndim", type=int, default=2)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--beta_teacher", type=float, default=1.0)
    # Agent
    parser.add_argument(
        "--agent",
        type=str,
        default="tb",
        choices=["tb", "gafntb", "teacher"],
    )
    parser.add_argument("--learn_pb", action="store_true")
    parser.add_argument("--eps", type=float, default=0.0)  # epsilon for exploration
    parser.add_argument("--logit_temp", type=float, default=1.0)  # logit temp for exploration-exploitation trade-off
    parser.add_argument("--n_hidden", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=2)

    # Buffer
    parser.add_argument("--use_buffer", action="store_true")
    parser.add_argument("--buffer_size", type=int, default=-1)
    parser.add_argument("--buffer_pri", type=str, default="none", choices=["none", "reward", "loss", "teacher_reward"])

    # Logger
    parser.add_argument("--n_logs", type=int, default=5)
    parser.add_argument("--logger", type=str, default="none", choices=["none", "wandb"])

    # Trainer
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--tb_z_lr", type=float, default=1e-1)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--n_train_steps", type=int, default=-1)  # -1 for automatic setting
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--ts_ratio", type=float, default=1)
    parser.add_argument("--no_tqdm", action="store_true")
    parser.add_argument("--num_empirical_loss", type=int, default=-1)  # -1 for automatic setting

    # GAFN
    parser.add_argument(
        "--no_edge_ri", action="store_true", help="Do not use edge-based intrinsic reward for training"
    )
    parser.add_argument("--ri_scale", type=float, default=0.01)  # 0.005 or 0.001 is used in the GAFN paper
    parser.add_argument("--ri_loss_weight", type=float, default=1.0)
    # RND params ... when agent == "gafntb"
    parser.add_argument("--rnd_hidden_dim", type=int, default=256)
    parser.add_argument("--rnd_latent_dim", type=int, default=128)

    # Teacher params
    parser.add_argument("--log", action="store_true", default=True)
    parser.add_argument("--no_log", action="store_false", dest="log")
    parser.add_argument("--C", type=float, default=19.0)
    parser.add_argument("--alpha", type=float, default=0.0)  # reward mixing coefficient

    # Back&Forth Local Search
    parser.add_argument("--ls", action="store_true", default=False)
    parser.add_argument("--ls_back_ratio", type=float, default=0.5)

    # Evaluation
    parser.add_argument("--eval_num_empirical_loss", type=int, default=100000)
    parser.add_argument("--eval_batch_size", type=int, default=1000)
    parser.add_argument("--plot", action="store_true")

    args = parser.parse_args()

    if int(args.n_train_steps) == -1:
        if args.ndim == 2:
            args.n_train_steps = 4000 if args.horizon <= 256 else 8000
        elif args.ndim == 4:
            args.n_train_steps = 4000 if args.horizon <= 16 else 16000
        else:
            args.n_train_steps = 4000

        if not args.ls:
            args.n_train_steps = int(args.n_train_steps * 1.5) 
        if args.use_buffer:
            args.n_train_steps = int(args.n_train_steps * 2.0)

    if int(args.num_empirical_loss) == -1:
        if args.ndim == 2:
            args.num_empirical_loss = 10000 if args.horizon <= 256 else 20000
        elif args.ndim == 4:
            args.num_empirical_loss = 10000 if args.horizon <= 16 else 20000
        else:
            args.num_empirical_loss = 20000

    if int(args.eval_num_empirical_loss) == -1:
        args.eval_num_empirical_loss = args.num_empirical_loss

    if args.plot:
        assert args.logger == "wandb", "plot option is only available when logger is wandb"

    # Set seed
    seed_everything(args.seed)

    # Set device
    device = torch.device(f"cuda:{args.device_num}" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    env_params = {
        "horizon": args.horizon,
        "ndim": args.ndim,
        "beta": args.beta,
    }
    env = DeceptiveGridWorld(**env_params)

    agent_common_params = {
        "horizon": args.horizon,
        "ndim": args.ndim,
        "eps": args.eps,
        "logit_temp": args.logit_temp,
        "device": device,
        "n_hidden": args.n_hidden,
        "n_layers": args.n_layers,
    }
    rnd_params = (
        {
            # state_dim is the dimension of the observation, which will be automatically set
            "hidden_dim": args.rnd_hidden_dim,
            "latent_dim": args.rnd_latent_dim,
        }
        if args.agent == "gafntb"
        else None
    )
    teacher_params = {"learn_pb": args.learn_pb, "beta_teacher": args.beta_teacher}
    agent_dict = {
        "tb": (TBAgent, {"learn_pb": args.learn_pb}),
        "teacher": (TeacherStudentTBAgent, teacher_params),
        "gafntb": (
            GAFNTBAgent,
            {
                "augmented": True,
                "rnd_params": rnd_params,
                "ri_scale": args.ri_scale,
                "ri_loss_weight": args.ri_loss_weight,
                "no_edge_ri": args.no_edge_ri,
                "learn_pb": args.learn_pb,
            },
        ),
    }

    agent_class, agent_param = agent_dict[args.agent]
    agent = agent_class(
        envs=[DeceptiveGridWorld(**env_params) for _ in range(max(args.batch_size, args.eval_batch_size))],
        **agent_common_params,
        **agent_param,
    )

    lr_dict = {
        "model": args.lr,
        "tb_z": args.tb_z_lr,
        "rnd": args.lr,  # only for GAFN
    }

    trainer_params = {
        "lr_dict": lr_dict,
        "clip_grad_norm": args.clip_grad_norm,
        "n_train_steps": args.n_train_steps,
        "batch_size": args.batch_size,
        "ts_ratio": args.ts_ratio,
        "ls": args.ls,
        "ls_back_ratio": args.ls_back_ratio,
        "log": args.log,
        "C": args.C,
        "alpha": args.alpha,
        "no_tqdm": args.no_tqdm,
        "num_empirical_loss": args.num_empirical_loss,
        "eval_num_empirical_loss": args.eval_num_empirical_loss,
        "eval_batch_size": args.eval_batch_size,
        "plot": args.plot,
        "use_buffer": args.use_buffer,
        "buffer_params": {
            "buffer_size": (
                args.buffer_size
                if args.buffer_size > 0 else
                int(0.1 * (args.horizon - 1 + args.ndim) * ((args.horizon - 1) ** (args.ndim - 1)))
            ),
            "prioritized": args.buffer_pri,
        },
        "n_logs": args.n_logs,
        "run_name": args.run_name,
    }

    logging_fn = None
    if args.logger == "wandb":
        import wandb

        wandb.init(
            project=f"TeacherGFN-Grid",
            name=f"{args.agent}{'_'+args.run_name if args.run_name else ''}",
            tags=[args.agent, f"h{args.horizon}ndim{args.ndim}"],
            settings=wandb.Settings(init_timeout=300),
        )
        for config in [env_params, agent_common_params, agent_param, trainer_params]:
            wandb.config.update(config)
        wandb.config.update(
            {"env_param_str": f"h{args.horizon}ndim{args.ndim}", "agent": args.agent, "run_name": args.run_name, "seed": args.seed}
        )
        # wandb.watch(agent)
        logging_fn = wandb.log

    trainer = Trainer(env, agent, logging_fn=logging_fn, **trainer_params)
    trainer.train()
    if args.logger == "wandb":
        wandb.finish()
