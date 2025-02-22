import io
import random
import math
from collections import defaultdict
from typing import Optional, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from PIL import Image
from tqdm import tqdm


def seed_everything(seed: Optional[int] = None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def compute_empirical_distribution_error(env, visited: List[Tuple]) -> Tuple[float, float]:
    if not len(visited):
        return 100, 100

    hist = defaultdict(int)
    for s in visited:
        hist[s] += 1

    unnormalized_density, true_density, end_states, state_to_idx = env.get_true_density()
    # Create a mapping from end_state to its index
    
    Z = len(visited)
    estimated_density = np.zeros_like(true_density)
    assert len(estimated_density) == len(end_states)

    # Update estimated_density
    for s in tqdm(visited):
        idx = state_to_idx.get(s)
        if idx is not None:
            estimated_density[idx] += 1
    estimated_density = estimated_density / Z

    # L1 distance
    l1 = abs(estimated_density - true_density).mean()
    # weighted L1 distance (weighted by unnormalized density)
    weighted_l1 = (abs(estimated_density - true_density) * unnormalized_density).mean()
    # # KL divergence
    # kl = (true_density * np.log(estimated_density / true_density)).sum()
    del estimated_density, hist
    return l1, weighted_l1


def plot_visit_counts(env, all_visited, this_visited, eval_visited, logging_fn=None, step=None):
    assert env.all_rewards is not None
    visit_count = np.zeros_like(env.all_rewards).reshape(*(env.horizon,) * env.ndim)
    for s in all_visited:
        visit_count[s] += 1

    this_visit_count = np.zeros_like(visit_count)
    for s in this_visited:
        this_visit_count[s] += 1

    imagined_visit_count = np.zeros_like(visit_count)
    for s in eval_visited:
        imagined_visit_count[s] += 1

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    for dat, ax in zip([visit_count, this_visit_count, imagined_visit_count], axes):
        if dat.ndim != 2:
            if dat.ndim % 2 == 0:
                power = dat.ndim // 2
                dat = dat.reshape(*(env.horizon**power,) * 2)
            else:
                raise ValueError(f"array.ndim should be even number, but {dat.ndim}")
        img = ax.imshow(dat, origin="lower")
        fig.colorbar(img, ax=ax)
    if logging_fn is not None:
        image = wandb.Image(fig)
        logging_fn({"all_vs_this_vs_imagined": image}, step=step)
    plt.close(fig)


def plot_empirical_distribution(env, visited, label="empirical_density", logging_fn=None, step=None, vmax=None):
    assert env.all_rewards is not None
    empirical_density = np.zeros_like(env.all_rewards).reshape(*(env.horizon,) * env.ndim)
    for s in visited:
        empirical_density[s] += 1
    empirical_density = empirical_density / empirical_density.sum()

    vmax = vmax if vmax is not None else max(0.0, empirical_density.max())

    plot_fn(env, empirical_density, label=label, vmax=vmax, logging_fn=logging_fn, step=step, colorbar=False, draw_axes=False)


def plot_target_distribution(env, vmax=None, logging_fn=None):
    assert env.all_rewards is not None
    target_map = env.all_rewards.reshape(*(env.horizon,) * env.ndim)
    target_map = target_map / target_map.sum()

    vmax = vmax if vmax is not None else max(0.0, target_map.max())

    plot_fn(env, target_map, label=f"target_density", vmax=vmax, logging_fn=logging_fn, step=0, colorbar=False, draw_axes=False)


def plot_fn(env, array, label, vmax, logging_fn=None, step=None, show=False, colorbar=False, draw_axes=False):
    if array.ndim != 2:
        if array.ndim % 2 == 0:
            power = array.ndim // 2
            array = array.reshape(*(env.horizon**power,) * 2)
        else:
            raise ValueError(f"array.ndim should be even number, but {array.ndim}")

    # fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    fig = plt.figure()
    fig.set_size_inches(1. * array.shape[0] / array.shape[1], 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    if not draw_axes:
        ax.set_axis_off()
    fig.add_axes(ax)
    img = ax.imshow((np.log(array + 0.0001)), origin="lower", vmax=math.log(vmax + 0.0001), interpolation="none")
    if colorbar:
        plt.colorbar(img, ax=ax)
    if logging_fn is not None:
        logging_fn({label: wandb.Image(fig)}, step=step)
    if show:
        plt.show()

    plt.close(fig)