"""
GridWorld environment
"""

from collections import deque
import gc
from typing import List, Optional, Tuple, Union, cast

import numpy as np


class DeceptiveGridWorld:
    """
    DeceptiveGridWorld; modified from https://github.com/ling-pan/GAFN
    """

    def __init__(self, horizon: int, ndim: int, beta: float) -> None:
        self.horizon = horizon
        self.ndim = ndim
        self.beta = beta

        # To be filled in get_true_density
        self.true_density_info = None
        self.all_rewards = None
        self.goals = None

    def reward_func(self, state: np.ndarray) -> np.ndarray:
        normalized_x = state / (self.horizon - 1) * 2 - 1
        abs_x = np.abs(normalized_x)

        rewards = (
            (abs_x > 0.2).prod(-1) * (-1e-1)
            + ((abs_x < 0.8) * (abs_x > 0.6)).prod(-1) * 2
            + 1e-1 + 1e-5
        )

        rewards = rewards ** self.beta

        return rewards

    def obs(self, state: Optional[np.ndarray] = None) -> np.ndarray:
        # state: (ndim,) with dtype=np.int32
        state = self.state if state is None else state
        z = np.zeros((self.horizon * self.ndim), dtype=np.float32)
        z[np.arange(self.ndim) * self.horizon + state] = 1
        return z

    def reset(self) -> Tuple[np.ndarray, np.ndarray, bool, np.ndarray]:
        self.state = np.zeros(self.ndim, dtype=np.int32)
        return self.obs(self.state), self.reward_func(self.state), False, self.state

    def step(
        self, action: int, state: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Union[float, np.ndarray], bool, np.ndarray]:
        s = cast(np.ndarray, state if state is not None else self.state)

        if action < self.ndim:
            s[action] += 1
        done = s.max() >= self.horizon - 1 or action == self.ndim
        return self.obs(s), 0 if not done else self.reward_func(s), done, s

    def parent_transitions(
        self, state: np.ndarray, used_stop_action: bool = False
    ) -> Tuple[List[np.ndarray], List[int]]:
        if used_stop_action:
            return [self.obs(state)], [self.ndim]

        parents = []
        actions = []
        for i in range(self.ndim):
            if state[i] > 0:
                state_parent = state.copy()
                state_parent[i] -= 1
                if state_parent.max() == self.horizon - 1:  # can't have a terminal parent
                    continue
                parents += [self.obs(state_parent)]  # Note that parents are saved as obs
                actions += [i]
        return parents, actions

    def get_true_density(self, beta=1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        # These calculation doesn't have to be done on GPU
        if self.true_density_info is not None:
            return self.true_density_info

        all_int_states = np.indices((self.horizon,)*self.ndim, dtype=np.int16).reshape(self.ndim, -1).T
        self.all_rewards = self.reward_func(all_int_states)
        reward_max = self.all_rewards.max()
        self.goals = list(map(tuple, all_int_states[self.all_rewards >= reward_max]))

        state_mask = (all_int_states == self.horizon - 1).sum(1) < 2

        unnormalized_density = self.all_rewards[state_mask] ** beta
        end_states = all_int_states[state_mask]
        true_density_info = (
            unnormalized_density,  # unnormalized density
            unnormalized_density / unnormalized_density.sum(),  # normalized density
            end_states,  # keys (states)
            {tuple(state): idx for idx, state in enumerate(end_states)},  # state to index
        )
        self.true_density_info = true_density_info
        del all_int_states
        gc.collect()
        return true_density_info

    def generate_backward(self, x: np.ndarray):
        curr_s = x.copy()
        curr_obs = self.obs(curr_s)

        # Now we work backward from that last transition
        obs_lst = deque()
        s_lst = deque()
        a_lst = deque()

        if curr_s.max() < self.horizon - 1:  # terminated with terminating action
            obs_lst.append(curr_obs)
            s_lst.append(curr_s.copy())
            a_lst.append([self.ndim])

        while curr_s.sum() > 0:  # decrease the curr_s's elements until it becomes all zeros
            obs_lst.appendleft(self.obs(curr_s))
            s_lst.appendleft(curr_s.copy())

            parents, actions = self.parent_transitions(curr_s, False)  # terminating action is already considered
            # Randomly choose a parent
            i = np.random.randint(0, len(parents))
            a = actions[i]
            curr_s[a] -= 1
            a_lst.appendleft([a])

        obs_lst.appendleft(self.obs(curr_s))
        s_lst.appendleft(curr_s.copy())

        obs_arr = np.stack(obs_lst)
        s_arr = np.stack(s_lst)
        a_arr = np.stack(a_lst)

        return obs_arr, s_arr, a_arr


if __name__ == "__main__":
    env = DeceptiveGridWorld(horizon=16, ndim=8, beta=1.0)
    actions = [0, 1, 0, 1, 0, 1, 0, 1]
    _obs, _r, _d, _s = env.reset()
    for a in actions:
        _obs, _r, _d, _s = env.step(a)
        print(_s)
    
    # Generate backward
    obs_arr, s_arr, a_arr = env.generate_backward(_s)
    print(s_arr)
