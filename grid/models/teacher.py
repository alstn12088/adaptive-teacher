"""
Model for the GridWorld environment
"""

from typing import Dict, List, Tuple, Union, cast
import math

import numpy as np
import math
import torch
from torch import nn

from grid.env import DeceptiveGridWorld
from grid.models.mlp import MLP
from grid.models.base import BaseAgent


def get_teacher_reward(
    discrepancies: Union[List[float], List[torch.Tensor]],
    train_batch: List[Tuple],
    log: bool = True,
    C: float = 19.0,
    alpha: float = 1.0,
) -> Union[List[float], List[torch.Tensor]]:
    if isinstance(discrepancies[0], torch.Tensor):
        teacher_rewards = []
        for _d, _tup in zip(discrepancies, train_batch):
            _d = cast(torch.Tensor, _d)
            _r = torch.ones_like(_d)
            _r[-1] += _tup[3]  # reward
            _a_r = (_d) ** 2 * (1 + torch.where(_d > 0, C, 0.0))
            if log:
                teacher_rewards.append((1 + 1e-3 + _a_r).log() * (_r ** alpha))
            else:
                teacher_rewards.append(_a_r * (_r ** alpha))

    else:  # List[float]
        maybe_log = lambda x: math.log(1 + 1e-3 + x) if log else x
        teacher_rewards = [
            maybe_log(_d ** 2 * (1 + int(_d > 0) * C)) * (_tup[3] ** alpha)
            for _d, _tup in zip(discrepancies, train_batch)
        ]
    return teacher_rewards




class TeacherStudentTBAgent(BaseAgent):
    """
    Teacher-Student agent based on Trajectory Balance

    There are two TB agents; one is the teacher model and the other is the student model.
    The student model is trained to sample proportionally to the target rewards, while 
    the teacher model is trained to sample proportionally to some function of the student's
    training loss.
    """

    def __init__(
        self,
        envs: List[DeceptiveGridWorld],
        horizon: int,
        ndim: int,
        eps: float = 0.1,
        logit_temp: float = 1.0,
        device: Union[torch.device, str] = "cpu",
        n_hidden: int = 256,
        n_layers: int = 2,
        learn_pb: bool = True,
        beta_teacher: float = 1.0,
    ) -> None:
        super().__init__(envs, horizon, ndim, eps, logit_temp, device)
        
        out_dim = 2 * ndim + 1  # x 2 for forward/backward, + 1 for terminating action
        self.model = MLP(dim_list=[horizon * ndim] + [n_hidden] * n_layers + [out_dim])
        self.model.to(device)
        self.teacher_model = MLP(dim_list=[horizon * ndim] + [n_hidden] * n_layers + [out_dim])
        self.teacher_model.to(device)
        self.log_Z = nn.parameter.Parameter(torch.zeros((), device=device, requires_grad=True))
        self.log_Z_teacher = nn.parameter.Parameter(torch.zeros((), device=device, requires_grad=True))
        self.learn_pb = learn_pb

        self.beta_teacher = beta_teacher

    def sample_many(self, batch_size: int, eval: bool = False, is_teacher: bool = False) -> Tuple[List[Tuple], List[Tuple]]:
        if batch_size == 0:
            return [], []

        assert batch_size <= len(self.envs)

        policy = self.teacher_model if is_teacher else self.model

        # Save trajectory
        batch_traj_obs, batch_traj_s, batch_traj_a = (
            [[] for _ in range(batch_size)],
            [[] for _ in range(batch_size)],
            [[] for _ in range(batch_size)],
        )

        # Save step results
        batch_obs, batch_r, batch_done, batch_s = [list(x) for x in zip(*[e.reset() for e in self.envs[:batch_size]])]
        batch_obs = self.to_ft(batch_obs)
        batch_s = self.to_lt(batch_s)

        # visited_x = []
        visited_x = [None] * batch_size
        not_done_indices = [i for i in range(batch_size)]

        while not all(batch_done):
            with torch.no_grad():
                pred = cast(torch.Tensor, policy(batch_obs))
                f_probs = (pred[..., : self.ndim + 1]).softmax(1)
                eps = 0.0 if eval else self.eps
                f_probs = (1 - eps) * f_probs + (eps) / (self.ndim + 1)
                actions = f_probs.multinomial(1)

            batch_step = [self.envs[i].step(int(a)) for i, a in zip(not_done_indices, actions)]

            for i, (curr_obs, curr_s, curr_a) in enumerate(zip(batch_obs, batch_s, actions)):
                env_idx = not_done_indices[i]
                batch_traj_obs[env_idx].append(curr_obs)
                batch_traj_s[env_idx].append(curr_s)
                batch_traj_a[env_idx].append(curr_a)

            new_batch_obs, new_batch_s, new_not_done_indices = [], [], []
            for i, (_obs, _r, _d, _s) in enumerate(batch_step):
                env_idx = not_done_indices[i]
                batch_done[env_idx] = _d

                if _d:
                    batch_r[env_idx] = _r  # episodic
                    batch_traj_obs[env_idx].append(self.to_ft(_obs))
                    batch_traj_s[env_idx].append(self.to_lt(_s))
                    # visited_x.append(tuple(_s))
                    visited_x[env_idx] = tuple(_s)
                else:
                    new_batch_obs.append(_obs)
                    new_batch_s.append(_s)
                    new_not_done_indices.append(env_idx)

            batch_obs, batch_s = self.to_ft(new_batch_obs), self.to_lt(new_batch_s)
            not_done_indices = new_not_done_indices

        batch_traj_obs_tsr = list(map(torch.stack, batch_traj_obs))
        batch_traj_s_tsr = list(map(torch.stack, batch_traj_s))
        batch_traj_a_tsr = list(map(torch.stack, batch_traj_a))

        train_batch = list(zip(batch_traj_obs_tsr, batch_traj_s_tsr, batch_traj_a_tsr, batch_r))

        return train_batch, visited_x

    def learn_from(
        self,
        train_batch: List[Tuple[torch.Tensor, ...]],
        is_teacher=False,
        new_rewards: Union[List[float], None] = None,
        return_logprob=False,
    ) -> Tuple[torch.Tensor, List[float], Dict]:
        inf = 1000000000

        if is_teacher:
            assert new_rewards is not None
            policy = self.teacher_model
            logZ = self.log_Z_teacher
            beta = self.beta_teacher
        else:
            policy = self.model
            logZ = self.log_Z
            beta = 1.0

        loss = torch.tensor(0.0, dtype=torch.float, device=self.device)
        discrepancies = []
        logit_lst = back_logit_lst = []

        for i, (_obs, _s, _a, _r) in enumerate(train_batch):
            _r = new_rewards[i] if new_rewards is not None else _r

            pred = cast(torch.Tensor, policy(_obs))

            edge_mask = torch.cat(
                [(_s == self.horizon - 1).float(), torch.zeros((_s.shape[0], 1), device=self.device)], 1
            )
            logits = (pred[..., : self.ndim + 1] - inf * edge_mask).log_softmax(1)

            init_edge_mask = (_s == 0).float()
            if self.learn_pb:
                back_logits = (pred[..., self.ndim + 1 :] - inf * init_edge_mask).log_softmax(1)
            else:
                back_logits = (- inf * init_edge_mask).log_softmax(1)

            logits = logits[:-1, :].gather(1, _a).squeeze(1)
            back_logits = (
                back_logits[1:-1, :].gather(1, _a[:-1, :]).squeeze(1)
                if _a[-1] == self.ndim
                else back_logits[1:, :].gather(1, _a).squeeze(1)
            )

            sum_logits = torch.sum(logits)
            sum_back_logits = torch.sum(back_logits)
            discrepancy = math.log(_r + 1e-8) * beta + sum_back_logits - logZ - sum_logits
            loss += discrepancy**2

            if not is_teacher:
                discrepancies.append(discrepancy.item())
            
            if return_logprob:
                logit_lst.append(sum_logits)
                back_logit_lst.append(sum_back_logits)

        info = {}
        if return_logprob:
            info["log_pfs"] = logit_lst
            info["log_pbs"] = back_logit_lst

        loss /= len(train_batch)
        return loss, discrepancies, info


class TeacherStudentDBAgent(BaseAgent):
    """
    Detailed Balance version of Teacher-Student agent
    """
    def __init__(
        self,
        envs: List[DeceptiveGridWorld],
        horizon: int,
        ndim: int,
        eps: float = 0.1,
        logit_temp: float = 1.0,
        device: Union[torch.device, str] = "cpu",
        n_hidden: int = 256,
        n_layers: int = 2,
        learn_pb: bool = True,
        beta_teacher: float = 1.0,
    ) -> None:
        super().__init__(envs, horizon, ndim, eps, logit_temp, device)

        out_dim = 2 * ndim + 2  # x 2 for forward/backward, + 1 for terminating action, + 1 for F(s)
        self.model = MLP(dim_list=[horizon * ndim] + [n_hidden] * n_layers + [out_dim])
        self.model.to(device)
        self.teacher_model = MLP(dim_list=[horizon * ndim] + [n_hidden] * n_layers + [out_dim])
        self.teacher_model.to(device)
        self.learn_pb = learn_pb

        self.beta_teacher = beta_teacher
    
    def sample_many(self, batch_size: int, eval: bool = False, is_teacher: bool = False) -> Tuple[List[Tuple[torch.Tensor, ...]], List[Tuple]]:
        if batch_size == 0:
            return [], []

        assert batch_size <= len(self.envs)

        policy = self.teacher_model if is_teacher else self.model

        # Save trajectory
        batch_traj_obs, batch_traj_s, batch_traj_a = (
            [[] for _ in range(batch_size)],
            [[] for _ in range(batch_size)],
            [[] for _ in range(batch_size)],
        )

        # Save step results
        batch_obs, batch_r, batch_done, batch_s = [list(x) for x in zip(*[e.reset() for e in self.envs[:batch_size]])]
        batch_obs = self.to_ft(batch_obs)
        batch_s = self.to_lt(batch_s)

        visited_x = []
        not_done_indices = [i for i in range(batch_size)]

        while not all(batch_done):
            with torch.no_grad():
                pred = cast(torch.Tensor, policy(batch_obs))
                f_probs = (pred[..., : self.ndim + 1]).softmax(1)
                eps = 0.0 if eval else self.eps
                f_probs = (1 - eps) * f_probs + (eps) / (self.ndim + 1)
                actions = f_probs.multinomial(1)

            batch_step = [self.envs[i].step(int(a)) for i, a in zip(not_done_indices, actions)]

            for i, (curr_obs, curr_s, curr_a) in enumerate(zip(batch_obs, batch_s, actions)):
                env_idx = not_done_indices[i]
                batch_traj_obs[env_idx].append(curr_obs)
                batch_traj_s[env_idx].append(curr_s)
                batch_traj_a[env_idx].append(curr_a)

            new_batch_obs, new_batch_s, new_not_done_indices = [], [], []
            for i, (_obs, _r, _d, _s) in enumerate(batch_step):
                env_idx = not_done_indices[i]
                batch_done[env_idx] = _d

                if _d:
                    batch_r[env_idx] = _r
                    batch_traj_obs[env_idx].append(self.to_ft(_obs))
                    batch_traj_s[env_idx].append(self.to_lt(_s))
                    visited_x.append(tuple(_s))
                else:
                    new_batch_obs.append(_obs)
                    new_batch_s.append(_s)
                    new_not_done_indices.append(env_idx)

            batch_obs, batch_s = self.to_ft(new_batch_obs), self.to_lt(new_batch_s)
            not_done_indices = new_not_done_indices

        batch_traj_obs_t = list(map(torch.stack, batch_traj_obs))
        batch_traj_s_t = list(map(torch.stack, batch_traj_s))
        batch_traj_a_t = list(map(torch.stack, batch_traj_a))

        train_batch = list(zip(batch_traj_obs_t, batch_traj_s_t, batch_traj_a_t, batch_r))

        return train_batch, visited_x

    def learn_from(
        self,
        train_batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]],
        is_teacher=False,
        new_rewards: Union[List[torch.Tensor], None] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Dict]:
        inf = 1000000000

        if is_teacher:
            assert new_rewards is not None
            policy = self.teacher_model
            beta = self.beta_teacher
        else:
            policy = self.model
            beta = 1.0

        loss = torch.tensor(0.0, dtype=torch.float, device=self.device)
        discrepancies = []  # This contains tensor of discrepancies from each transition

        for i, (_obs, _s, _a, _r) in enumerate(train_batch):
            _r = new_rewards[i].sum() if new_rewards is not None else self.to_ft(_r)

            pred = cast(torch.Tensor, policy(_obs))

            edge_mask = torch.cat(
                [(_s == self.horizon - 1).float(), torch.zeros((_s.shape[0], 1), device=self.device)], 1
            )
            logits = (pred[..., : self.ndim + 1] - inf * edge_mask).log_softmax(1)

            init_edge_mask = (_s == 0).float()
            if self.learn_pb:
                back_logits = (pred[..., self.ndim + 1 : -1] - inf * init_edge_mask).log_softmax(1)
            else:
                back_logits = (- inf * init_edge_mask).log_softmax(1)

            logits = logits[:-1, :].gather(1, _a).squeeze(1)

            back_logits = (
                back_logits[1:-1, :].gather(1, _a[:-1, :]).squeeze(1)
                if _a[-1] == self.ndim
                else back_logits[1:, :].gather(1, _a).squeeze(1)  # when it reaches the end of the grid
            )

            if _a[-1] == self.ndim:  # backward probability for the terminating action is 1
                back_logits = torch.cat([back_logits, torch.zeros((1,), device=self.device)], 0)

            log_Fs_start = pred[:-1, -1]
            log_Fs_end = pred[1:-1, -1]
            log_Fs_end = torch.cat([log_Fs_end, (_r + 1e-8).unsqueeze(0).log() * beta], 0)
            discrepancy = log_Fs_start - log_Fs_end + logits - back_logits

            loss += discrepancy.pow(2).sum()

            if not is_teacher:
                discrepancies.append(discrepancy.detach())

        loss /= len(train_batch)
        return loss, discrepancies, {}

    def learn_from_transition(
        self,
        buffer_batch: List[Tuple[Tuple[int], int, Tuple[int], float, bool]],
        is_teacher=False,
        new_rewards: Union[List[torch.Tensor], None] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Dict]:
        inf = 1000000000

        if is_teacher:
            assert new_rewards is not None
            policy = self.teacher_model
            beta = self.beta_teacher
        else:
            policy = self.model
            beta = 1.0

        loss = torch.tensor(0.0, dtype=torch.float, device=self.device)
        discrepancies = []

        for i, (_s, _a, _s_next, _r, _is_terminal) in enumerate(buffer_batch):
            _r = new_rewards[i].sum() if new_rewards is not None else self.to_ft(_r)

            _s_arr, _s_next_arr = np.array(_s), np.array(_s_next)
            _obs = self.envs[0].obs(_s_arr)
            _obs_next = self.envs[0].obs(_s_next_arr)
            _obs_tsr = self.to_ft(np.stack([_obs, _obs_next]))  # (2, dim*horizon)
            _s_tsr = self.to_lt(np.stack([_s_arr, _s_next_arr]))  # (2, dim)
            _a_tsr = self.to_lt(np.array([[_a]]))  # (1, 1)
            pred = cast(torch.Tensor, policy(_obs_tsr))

            edge_mask = torch.cat(
                [(_s_tsr == self.horizon - 1).float(), torch.zeros((_s_tsr.shape[0], 1), device=self.device)], 1
            )
            logits = (pred[..., : self.ndim + 1] - inf * edge_mask).log_softmax(1)

            init_edge_mask = (_s_tsr == 0).float()
            if self.learn_pb:
                back_logits = (pred[..., self.ndim + 1 : -1] - inf * init_edge_mask).log_softmax(1)
            else:
                back_logits = (- inf * init_edge_mask).log_softmax(1)

            logits = logits[:-1, :].gather(1, _a_tsr).squeeze(1)

            back_logits = (
                back_logits[1:-1, :].gather(1, _a_tsr[:-1, :]).squeeze(1)
                if _a_tsr[-1] == self.ndim
                else back_logits[1:, :].gather(1, _a_tsr).squeeze(1)  # when it reaches the end of the grid
            )

            if _a_tsr[-1] == self.ndim:  # backward probability for the terminating action is 1
                back_logits = torch.cat([back_logits, torch.zeros((1,), device=self.device)], 0)

            log_Fs_start = pred[:-1, -1]
            if _is_terminal:
                log_Fs_end = pred[1:-1, -1]
                log_Fs_end = torch.cat([log_Fs_end, (_r + 1e-8).unsqueeze(0).log() * beta], 0)
            else:
                log_Fs_end = pred[1:, -1]
            discrepancy = log_Fs_start - log_Fs_end + logits - back_logits

            loss += discrepancy.pow(2).sum()

            if not is_teacher:
                discrepancies.append(discrepancy.detach())

        loss /= len(buffer_batch)
        return loss, discrepancies, {}
