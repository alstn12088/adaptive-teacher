
from typing import List, Tuple, cast

import torch
from torch import nn

from grid.models import BaseAgent, TeacherStudentDBAgent, TeacherStudentTBAgent
from grid.models.teacher import get_teacher_reward


def back_and_forth(
    agent: BaseAgent,
    train_batch: List[Tuple],
    ls_back_ratio=0.5,
    iterations=8,
    eps_noisy=False,
    teacher_ls=False,
    log=True,
    C=19.0,
    alpha=0.0,
) -> Tuple[List[Tuple], List[Tuple]]:
    # Back & forth local search
    policy = cast(nn.Module, agent.model)
    if teacher_ls:
        assert isinstance(agent, (TeacherStudentDBAgent, TeacherStudentTBAgent))
        policy = agent.teacher_model
        with torch.no_grad():
            discrepancies = agent.learn_from(train_batch, is_teacher=False)[1]  # evaluate with the main model
            teacher_rewards = get_teacher_reward(discrepancies, train_batch, log=log, C=C, alpha=alpha)

    for _ in range(iterations):
        new_train_batch = []
        for i, (traj_obs, traj_s, traj_a, reward) in enumerate(train_batch):
            n_back_steps = int(traj_s.shape[0] * ls_back_ratio)
            if n_back_steps == 0:
                new_train_batch.append((traj_obs, traj_s, traj_a, reward))
                continue

            back_arrs = agent.envs[0].generate_backward(traj_s[-1].cpu().numpy())

            source_to_mid = [torch.from_numpy(arr[:-n_back_steps]) for arr in back_arrs]
            obs_last, s_last = source_to_mid[0][-1], source_to_mid[1][-1]

            mid_to_target_a, mid_to_target_obs, mid_to_target_s = [], [], []
            done = False
            while not done:
                with torch.no_grad():
                    pred = cast(torch.Tensor, policy(obs_last.unsqueeze(0)))
                    f_probs = (pred[..., : agent.ndim + 1]).softmax(1)
                    eps = 0.0 if not eps_noisy else agent.eps
                    f_probs = (1 - eps) * f_probs + (eps) / (agent.ndim + 1)
                    actions = f_probs.multinomial(1)

                new_obs, new_r, done, new_s = agent.envs[i].step(int(actions[0]), s_last.numpy().copy())
                obs_last = agent.to_ft(new_obs)
                s_last = agent.to_lt(new_s)

                mid_to_target_a.append(actions[0])
                mid_to_target_obs.append(obs_last)
                mid_to_target_s.append(s_last)

            # Replace the trajectory
            traj_obs = torch.cat([source_to_mid[0], torch.stack(mid_to_target_obs)], dim=0)
            traj_s = torch.cat([source_to_mid[1], torch.stack(mid_to_target_s)], dim=0)
            traj_a = torch.cat([source_to_mid[2], torch.stack(mid_to_target_a)], dim=0)
            new_train_batch.append((traj_obs, traj_s, traj_a, new_r))

        # Selection
        if not teacher_ls:
            train_batch = [
                _new if _new[3] > _old[3] else _old for _new, _old in zip(new_train_batch, train_batch)
            ]
        else:  # use the teacher_rewards
            assert isinstance(agent, (TeacherStudentDBAgent, TeacherStudentTBAgent))
            # Re-evaluate the teacher_reward
            with torch.no_grad():
                new_discrepancies = agent.learn_from(new_train_batch, is_teacher=False)[1]  # evaluate with the main model
                new_teacher_rewards = get_teacher_reward(new_discrepancies, new_train_batch, log=log, C=C, alpha=alpha)

            replaced = [True if new_teacher_rewards[i] > teacher_rewards[i] else False for i in range(len(train_batch))]
            train_batch = [
                _new if replaced[i] else _old for i, (_new, _old) in enumerate(zip(new_train_batch, train_batch))
            ]
            teacher_rewards = [new_teacher_rewards[i] if replaced[i] else teacher_rewards[i] for i in range(len(train_batch))]

    visited_x = [tuple(tup[1][-1].tolist()) for tup in train_batch]
    return train_batch, visited_x
