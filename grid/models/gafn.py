"""
Model for the GridWorld environment, modified for GAFN
"""

from typing import Any, Dict, List, Optional, Tuple, Union, cast

import torch
from torch import nn

from grid.env import DeceptiveGridWorld
from grid.models import BaseAgent
from grid.models.mlp import MLP


class RND(nn.Module):
    """
    Random Network Distillation (RND) implementation for GAFN in GridWorld environment
    reference for GAFN: https://arxiv.org/abs/2210.03308
    """
    def __init__(self, state_dim, hidden_dim=256, latent_dim=128):
        super(RND, self).__init__()

        self.random_target_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.predictor_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, next_state):
        random_phi_s_next = self.random_target_network(next_state)
        predicted_phi_s_next = self.predictor_network(next_state)
        return random_phi_s_next, predicted_phi_s_next

    def compute_intrinsic_reward(self, next_states):
        random_phi_s_next, predicted_phi_s_next = self.forward(next_states)

        intrinsic_reward = torch.norm(predicted_phi_s_next.detach() - random_phi_s_next.detach(), dim=-1, p=2)  # type: ignore

        return intrinsic_reward

    def compute_loss(self, next_states):
        random_phi_s_next, predicted_phi_s_next = self.forward(next_states)
        rnd_loss = torch.norm(predicted_phi_s_next - random_phi_s_next.detach(), dim=-1, p=2)  # type: ignore
        mean_rnd_loss = torch.mean(rnd_loss)
        return mean_rnd_loss


class GAFNTBAgent(BaseAgent):
    def __init__(
        self,
        envs: List[DeceptiveGridWorld],
        horizon: int,
        ndim: int,
        eps: float = 0.0,
        logit_temp: float = 1.0,
        device: Union[torch.device, str] = "cpu",
        n_hidden: int = 256,
        n_layers: int = 2,
        learn_pb: bool = True,
        augmented: bool = True,
        ri_scale: float = 0.5,
        ri_loss_weight: float = 1.0,
        rnd_params: Optional[dict] = None,
        no_edge_ri: bool = False,
    ) -> None:
        super().__init__(envs, horizon, ndim, eps, logit_temp, device)

        out_dim = 2 * ndim + 1 + (1 if augmented else 0)
        self.model = MLP(dim_list=[horizon * ndim] + [n_hidden] * n_layers + [out_dim])
        self.model.to(device)

        self.log_Z = nn.parameter.Parameter(torch.zeros((), device=device, requires_grad=True))
        self.learn_pb = learn_pb

        self.augmented = augmented

        self.ri_model = RND(state_dim=horizon * ndim, **(rnd_params or {}))
        self.ri_model.to(device)
        self.ri_scale = ri_scale
        self.ri_loss_weight = ri_loss_weight

        self.no_edge_ri = no_edge_ri

    def sample_many(
            self, batch_size: int, eval=False, beta=1.0,
        ) -> Tuple[List[Tuple[torch.Tensor, ...]], List[Tuple]]:
        # replayed = self.replay.sample() if self.use_buffer and self.replay else []
        # TODO: Use the replay buffer

        assert batch_size <= len(self.envs)

        # Save trajectory
        batch_traj_obs, batch_traj_s, batch_traj_a = (
            [[] for _ in range(batch_size)],
            [[] for _ in range(batch_size)],
            [[] for _ in range(batch_size)],
        )

        # Save step results
        batch_obs, batch_r, batch_done, batch_s = [list(x) for x in zip(*[e.reset() for e in self.envs[:batch_size]])]
        batch_ri = [[] for _ in range(batch_size)]  # for intrinsic reward, if not augmented, this is empty
        batch_obs = self.to_ft(batch_obs)
        batch_s = self.to_lt(batch_s)

        # visited_x = []
        visited_x = [None] * batch_size
        not_done_indices = [i for i in range(batch_size)]
        while not all(batch_done):
            with torch.no_grad():
                pred = cast(torch.Tensor, self.model(batch_obs))
                logit_temp = 1.0 if eval else self.logit_temp
                f_probs = (pred[..., : self.ndim + 1] / logit_temp).softmax(1)
                eps = 0.0 if eval else self.eps
                f_probs = (1 - eps) * f_probs + (eps) / (self.ndim + 1)
                actions = f_probs.multinomial(1)

            batch_step = [self.envs[i].step(int(a)) for i, a in zip(not_done_indices, actions)]

            for i, (curr_obs, curr_s, curr_a) in enumerate(zip(batch_obs, batch_s, actions)):
                env_idx = not_done_indices[i]
                batch_traj_obs[env_idx].append(curr_obs)
                batch_traj_s[env_idx].append(curr_s)
                batch_traj_a[env_idx].append(curr_a)

            if self.augmented:
                next_obs = self.to_ft([step[0] for step in batch_step])
                ri = self.ri_model.compute_intrinsic_reward(next_obs) * self.ri_scale
                for i, _ri in enumerate(ri):
                    env_idx = not_done_indices[i]
                    batch_ri[env_idx].append(_ri)

            new_batch_obs, new_batch_s, new_not_done_indices = [], [], []
            for i, (_obs, _r, _d, _s) in enumerate(batch_step):
                env_idx = not_done_indices[i]
                batch_done[env_idx] = _d

                if _d:
                    batch_r[env_idx] = _r
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

        batch_traj_obs_t = list(map(torch.stack, batch_traj_obs))
        batch_traj_s_t = list(map(torch.stack, batch_traj_s))
        batch_traj_a_t = list(map(torch.stack, batch_traj_a))
        batch_r = list(map(self.to_ft, batch_r))

        batch_ri = list(map(self.to_ft, batch_ri))  # empty tensors if not augmented
        if self.augmented:
            assert len(batch_ri[0]) > 0  # This is ugly
            batch_r = [r + ri[-1] for r, ri in zip(batch_r, batch_ri)]

        train_batch = list(zip(batch_traj_obs_t, batch_traj_s_t, batch_traj_a_t, batch_r, batch_ri))

        return train_batch, visited_x

    def learn_from(self, train_batch: List[Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        inf = 1000000000

        loss = torch.tensor(0.0, dtype=torch.float, device=self.device)
        for _tup in train_batch:
            _obs, _s, _a, _r, _ri = _tup if self.augmented else (*_tup, torch.tensor([], device=self.device))

            pred = cast(torch.Tensor, self.model(_obs))

            edge_mask = torch.cat(
                [(_s == self.horizon - 1).float(), torch.zeros((_s.shape[0], 1), device=self.device)], 1
            )
            logits = (pred[..., : self.ndim + 1] - inf * edge_mask).log_softmax(1)

            init_edge_mask = (_s == 0).float()
            back_logits_end_pos = -1 if self.augmented else pred.shape[-1]
            init_edge_mask = (_s == 0).float()
            if self.learn_pb:
                back_logits = (pred[..., self.ndim + 1 : back_logits_end_pos] - inf * init_edge_mask).log_softmax(1)
            else:
                back_logits = (- inf * init_edge_mask).log_softmax(1)

            logits = logits[:-1, :].gather(1, _a).squeeze(1)
            back_logits = (
                back_logits[1:-1, :].gather(1, _a[:-1, :]).squeeze(1)
                if _a[-1] == self.ndim else
                back_logits[1:, :].gather(1, _a).squeeze(1)
            )

            sum_logits = torch.sum(logits)

            if self.augmented:
                flow = (pred[..., -1][1: -1]).exp()  # State Flow
                augmented_r_f = _ri[:-1] / flow if not self.no_edge_ri else 0  # _ri[:-1] to exclude the last state
                sum_back_logits = (
                    torch.sum((back_logits.exp() + augmented_r_f).log())
                    if _a[-1] == self.ndim else
                    torch.sum((back_logits[:-1].exp() + augmented_r_f).log()) + back_logits[-1]
                )
            else:
                sum_back_logits = torch.sum(back_logits)

            curr_ll_diff = self.log_Z + sum_logits - (_r + 1e-8).log() - sum_back_logits
            loss += curr_ll_diff**2

        loss /= len(train_batch)
        if self.augmented:
            ri_loss = torch.stack([self.ri_model.compute_loss(bat[0]) for bat in train_batch]).mean()
            loss += self.ri_loss_weight * ri_loss
        return loss, {}
