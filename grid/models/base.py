"""
Model for the GridWorld environment
"""

from abc import ABC, abstractmethod
import math
from typing import Any, Dict, List, Tuple, Union, cast
import itertools

import numpy as np
import torch
from torch import nn

from grid.env import DeceptiveGridWorld
from grid.models.mlp import MLP


class BaseAgent(nn.Module, ABC):
    def __init__(
        self,
        envs: List[DeceptiveGridWorld],
        horizon: int,
        ndim: int,
        eps: float = 0.1,
        logit_temp: float = 1.0,   
        device: Union[torch.device, str] = "cpu",
    ) -> None:
        super().__init__()
        self.envs = envs  # TODO: do not use list of envs but a single env
        self.horizon = horizon
        self.ndim = ndim
        self.eps = eps
        self.logit_temp = logit_temp
        self.device = device

        self.to_ft = lambda x: torch.tensor(np.array(x, dtype=np.float32), device=device)
        self.to_lt = lambda x: torch.tensor(np.array(x, dtype=np.int64), device=device)

    @abstractmethod
    def sample_many(self, batch_size: int, eval: bool = False, beta=1.0) -> Tuple[List[Tuple[torch.Tensor, ...]], List[Tuple]]:
        raise NotImplementedError

    @abstractmethod
    def learn_from(self, train_batch: List[Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        raise NotImplementedError


class FMAgent(BaseAgent):
    """
    Agent for the GFlowNet with Flow-Matching (FM) loss
    reference for GFlowNet and FM: https://arxiv.org/abs/2106.04399

    One parameterization:
        1. Edge-flow, F(s -> s')
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
    ) -> None:
        super().__init__(envs, horizon, ndim, eps, logit_temp, device)
        self.model = MLP(dim_list=[horizon * ndim] + [n_hidden] * n_layers + [ndim + 1])  # + 1 for terminating
        self.model.to(device)

    def sample_many(self, batch_size: int, eval: bool = False) -> Tuple[List[Tuple[torch.Tensor, ...]], List[Tuple]]:
        train_batch = []

        batch_obs = self.to_ft([e.reset()[0] for e in self.envs])
        batch_done = [False] * batch_size

        visited_x = []

        while not all(batch_done):
            # TODO: Batchify the envs
            batch_env = [self.envs[i] for i, d in enumerate(batch_done) if not d]
            with torch.no_grad():
                pred = cast(torch.Tensor, self.model(batch_obs))
                logit_temp = 1.0 if eval else self.logit_temp
                f_probs = (pred[..., : self.ndim + 1] / logit_temp).softmax(1)
                eps = 0.0 if eval else self.eps
                f_probs = (1 - eps) * f_probs + (eps) / (self.ndim + 1)
                actions = f_probs.multinomial(1)

            batch_step = [e.step(int(a)) for e, a in zip(batch_env, actions)]
            p_a = [
                self.envs[0].parent_transitions(sp_state, bool(a == self.ndim))
                for a, (_, _, _, sp_state) in zip(actions, batch_step)
            ]
            train_batch += [
                tuple(self.to_ft(x) for x in (p, a, [r], [sp], [d])) for (p, a), (sp, r, d, _) in zip(p_a, batch_step)
            ]

            c = itertools.count(0)
            m = {j: next(c) for j in range(batch_size) if not batch_done[j]}

            batch_done = [bool(d or batch_step[m[i]][2]) for i, d in enumerate(batch_done)]
            batch_obs = self.to_ft([st[0] for st in batch_step if not st[2]])

            for _, r, d, sp in batch_step:
                if d:
                    visited_x.append(tuple(sp))

        return train_batch, visited_x

    def learn_from(self, train_batch: List[Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        loginf = self.to_ft([1000])

        batch_idxs = self.to_lt(
            list(itertools.chain(*[[i] * len(parents) for i, (parents, _, _, _, _) in enumerate(train_batch)]))
        )

        parents, actions, r, sp, done = map(torch.cat, zip(*train_batch))

        parents_Qsa = self.model(parents)[torch.arange(parents.shape[0]), actions.long()]
        # Note: model outputs log-flow, so we exponentiate before summing

        in_flow = torch.log(
            torch.zeros((sp.shape[0],), device=self.device).index_add_(0, batch_idxs, torch.exp(parents_Qsa))
        )

        next_q = self.model(sp)

        next_qd = next_q * (1 - done).unsqueeze(1) + done.unsqueeze(1) * (-loginf)
        log_rewards = torch.log(r).unsqueeze(1)
        out_flow = torch.logsumexp(torch.cat([log_rewards, next_qd], 1), 1)
        loss = (in_flow - out_flow).pow(2).mean()

        with torch.no_grad():
            term_loss = ((in_flow - out_flow) * done).pow(2).sum() / (done.sum() + 1e-20)
            flow_loss = ((in_flow - out_flow) * (1 - done)).pow(2).sum() / ((1 - done).sum() + 1e-20)

        return loss, {"term_loss": term_loss, "flow_loss": flow_loss}


class TBAgent(BaseAgent):
    """
    Agent for the GFlowNet with Trajectory-Balance (TB) loss
    reference for TB: https://arxiv.org/abs/2201.13259

    Three parameterizations:
        1. Total-flow, Z
        2. Forward transition probability, P_F(s' | s)
        3. Backward transition probability, P_B(s | s') ... this can be fixed
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
    ) -> None:
        super().__init__(envs, horizon, ndim, eps, logit_temp, device)
        
        out_dim = 2 * ndim + 1  # x 2 for forward/backward, + 1 for terminating action
        self.model = MLP(dim_list=[horizon * ndim] + [n_hidden] * n_layers + [out_dim])
        self.model.to(device)
        self.log_Z = nn.parameter.Parameter(torch.zeros((), device=device, requires_grad=True))
        self.learn_pb = learn_pb

    def sample_many(self, batch_size: int, eval: bool = False, beta=1.0) -> Tuple[List[Tuple], List[Tuple]]:
        assert batch_size <= len(self.envs)

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
                pred = cast(torch.Tensor, self.model(batch_obs))
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

    def learn_from(self, train_batch: List[Tuple]) -> Tuple[torch.Tensor, List[float], Dict]:
        inf = 1000000000

        loss = torch.tensor(0.0, dtype=torch.float, device=self.device)
        discrepancies = []

        for _obs, _s, _a, _r in train_batch:
            pred = cast(torch.Tensor, self.model(_obs))

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

            discrepancy = math.log(_r + 1e-8) + sum_back_logits - self.log_Z - sum_logits
            loss += discrepancy**2

            discrepancies.append(discrepancy.item())

        loss /= len(train_batch)
        return loss, discrepancies, {}


class DBAgent(BaseAgent):
    """
    Agent for the GFlowNet with Detailed-Balance (DB) loss
    reference for DB: https://arxiv.org/abs/2111.09266

    Three parameterizations:
        1. State-flow, F(s)
        2. Forward transition probability, P_F(s' | s)
        3. Backward transition probability, P_B(s | s') ... this can be fixed
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
    ) -> None:
        super().__init__(envs, horizon, ndim, eps, logit_temp, device)

        out_dim = 2 * ndim + 2  # x 2 for forward/backward, + 1 for terminating action, + 1 for F(s)
        self.model = MLP(dim_list=[horizon * ndim] + [n_hidden] * n_layers + [out_dim])
        self.model.to(device)
        self.learn_pb = learn_pb

    def sample_many(self, batch_size: int, eval: bool = False) -> Tuple[List[Tuple[torch.Tensor, ...]], List[Tuple]]:
        if batch_size == 0:
            return [], []

        assert batch_size <= len(self.envs)

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
                pred = cast(torch.Tensor, self.model(batch_obs))
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
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Dict]:
        inf = 1000000000

        loss = torch.tensor(0.0, dtype=torch.float, device=self.device)
        discrepancies = []

        for _obs, _s, _a, _r in train_batch:
            pred = cast(torch.Tensor, self.model(_obs))

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
            log_Fs_end = torch.cat([log_Fs_end, (self.to_ft(_r) + 1e-8).unsqueeze(0).log()], 0)
            discrepancy_transition = log_Fs_start - log_Fs_end + logits - back_logits

            loss += discrepancy_transition.pow(2).sum()

            discrepancies.append(discrepancy_transition.detach())

        loss /= len(train_batch)
        return loss, discrepancies, {}

    def learn_from_transition(
        self, buffer_batch: List[Tuple[Tuple[int], int, Tuple[int], float, bool]]
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Dict]:
        inf = 1000000000

        loss = torch.tensor(0.0, dtype=torch.float, device=self.device)
        discrepancies = []

        # TODO: parallelize
        for _s, _a, _s_next, _r, _is_terminal in buffer_batch:
            _s_arr, _s_next_arr = np.array(_s), np.array(_s_next)
            _obs = self.envs[0].obs(_s_arr)
            _obs_next = self.envs[0].obs(_s_next_arr)
            _obs_tsr = self.to_ft(np.stack([_obs, _obs_next]))  # (2, dim*horizon)
            _s_tsr = self.to_lt(np.stack([_s_arr, _s_next_arr]))  # (2, dim)
            _a_tsr = self.to_lt(np.array([[_a]]))  # (1, 1)
            pred = cast(torch.Tensor, self.model(_obs_tsr))

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
                log_Fs_end = torch.cat([log_Fs_end, (self.to_ft(_r).unsqueeze(0) + 1e-8).log()], 0)
            else:
                log_Fs_end = pred[1:, -1]
            discrepancy = log_Fs_start - log_Fs_end + logits - back_logits

            loss += discrepancy.pow(2).sum()

            discrepancies.append(discrepancy.detach())

        loss /= len(buffer_batch)
        return loss, discrepancies, {}
