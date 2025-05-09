from itertools import chain
import numpy as np
import torch
from torch.distributions import Categorical
import wandb
from pathlib import Path
from tqdm import tqdm
import copy

from ..utils import tensor_to_np, batch, pack, unpack
from ..data import Experience
import math

class AugGFlowNet():
  """ Trajectory balance parameterization:
      logZ, forward policy, backward policy.
      Default behavior:
      - No parameter sharing between forward/backward policy
      - Separate optimizers for forward/backward policy; this is needed for
        guided TB. Logically unnecessary for regular TB, but equivalent to
        using a single optimizer for both policies.

      Forward and backward policy classes are specified by mdp.
  """
  def __init__(self, args, mdp, actor, rnd_target, rnd_predict):
    self.args = args
    self.mdp = mdp
    self.actor = actor

    self.policy_fwd = actor.policy_fwd
    self.policy_back = actor.policy_back
    self.flow_net = torch.nn.Linear(self.args.ssr_embed_dim, 1)


    self.rnd_target = rnd_target.policy_fwd
    self.rnd_predict = rnd_predict.policy_fwd


    self.logZ = torch.nn.Parameter(torch.tensor([5.], device=self.args.device))

    
    
    self.logZ_lower = 0

    self.nets = [self.policy_fwd, self.policy_back, self.rnd_target, self.rnd_predict, self.flow_net]
    for net in self.nets:
      net.to(args.device)

    self.clip_grad_norm_params = [self.policy_fwd.parameters(),
                                  self.policy_back.parameters(),
                                  self.rnd_predict.parameters(),
                                  self.flow_net.parameters()]

    self.optimizer_back = torch.optim.Adam([
        {
          'params': self.policy_back.parameters(),
          'lr': args.lr_policy
        }])
    self.optimizer_fwdZ = torch.optim.Adam([
        {
          'params': self.policy_fwd.parameters(),     
          'lr': args.lr_policy
        }, 
        {
          'params': self.logZ, 
          'lr': args.lr_z
        },
        {
            'params': self.flow_net.parameters(),
            'lr': args.lr_z
            }])
    self.optimizer_rnd = torch.optim.Adam([
        {
          'params': self.rnd_target.parameters(),
          'lr': args.lr_policy
        }
    ])

    self.optimizers = [self.optimizer_fwdZ, self.optimizer_back, self.optimizer_rnd]
    
    self.online_loss_step = 0
    self.offline_loss_step = 0
    pass
  
  """
    logZ
  """
  def init_logz(self, val):
    print(f'Initializing Z to {val}. Using this as floor for clamping ...')
    self.logZ.data = torch.tensor([val],
        device=self.args.device, requires_grad=True)
    assert self.logZ.is_leaf
    self.logZ_lower = val
    return

  def clamp_logZ(self):
    """ Clamp logZ to min value. Default assumes logZ > 0 (Z > 1). """
    self.logZ.data = torch.clamp(self.logZ, min=self.logZ_lower)
    return

  """
    Forward and backward policy
  """
  def fwd_logps_unique(self, batch):
    """ Differentiable; output logps of unique children/parents.
    
        See policy.py : logps_unique for more info.

        Input: List of [State], n items
        Returns
        -------
        state_to_logp: List of dicts mapping state to torch.tensor
    """
    return self.policy_fwd.logps_unique(batch)
  
  def fwd_logfs_unique(self, batch):
    return self.logF(batch)

  def fwd_sample(self, batch, epsilon=0.0):
    """ Non-differentiable; sample a child or parent.
    
        See policy.py : sample for more info.

        Input: batch: List of [State], or State
        Output: List of [State], or State
    """
    return self.policy_fwd.sample(batch, epsilon=epsilon)

  def back_logps_unique(self, batch):
    """ Differentiable; output logps of unique children/parents. """
    return self.policy_back.logps_unique(batch)
  
  def back_logfs_unique(self, batch):
    return self.logF(batch)

  def back_sample(self, batch):
    """ Non-differentiable; sample a child or parent.

        Input: batch: List of [State], or State
        Output: List of [State], or State
    """
    return self.policy_back.sample(batch)

  """
    Exploration & modified policies
  """
  def batch_fwd_sample(self, n, epsilon=0.0, uniform=False):
    """ Batch samples dataset with n items.

        Parameters
        ----------
        n: int, size of dataset.
        epsilon: Chance in [0, 1] of uniformly sampling a unique child.
        uniform: If true, overrides epsilon to 1.0
        unique: bool, whether all samples should be unique

        Returns
        -------
        dataset: List of [Experience]
    """
    print('Sampling dataset ...')
    if uniform:
      print('Using uniform forward policy on unique children ...')
      epsilon = 1.0
    incomplete_trajs = [[self.mdp.root()] for _ in range(n)]
    complete_trajs = []
    while len(incomplete_trajs) > 0:
      inp = [t[-1] for t in incomplete_trajs]
      samples = self.fwd_sample(inp, epsilon=epsilon)
      for i, sample in enumerate(samples):
        incomplete_trajs[i].append(sample)
      
      # Remove complete trajs that hit leaf
      temp_incomplete = []
      for t in incomplete_trajs:
        if not t[-1].is_leaf:
          temp_incomplete.append(t)
        else:
          complete_trajs.append(t)
      incomplete_trajs = temp_incomplete

    # convert trajs to exps
    list_exps = []
    for traj in complete_trajs:
      x = traj[-1]
      r = self.mdp.reward(x)
      # prevent NaN
      exp = Experience(traj=traj, x=x, r=r,
        logr=torch.nan_to_num(torch.log(torch.tensor(r, dtype=torch.float32)).to(device=self.args.device), neginf=-100.0))
      list_exps.append(exp)
    return list_exps
      
  def batch_back_sample(self, xs):
    """ Batch samples trajectories backwards from xs.
        Batches over xs, iteratively sampling parents for each x in parallel.
        Effective batch size decreases when some trajectories hit root early.

        Input xs: List of [State], or State
        Return trajs: List of List[State], or List[State]
    """
    batched = bool(type(xs) is list)
    if not batched:
      xs = [xs]

    complete_trajs = []
    incomplete_trajs = [[x] for x in xs]
    while len(incomplete_trajs) > 0:
      inp = [t[0] for t in incomplete_trajs]
      samples = self.back_sample(inp)
      for i, sample in enumerate(samples):
        incomplete_trajs[i].insert(0, sample)
      
      # Remove complete trajectories that hit root
      temp_incomplete = []
      for t in incomplete_trajs:
        if t[0] != self.mdp.root():
          temp_incomplete.append(t)
        else:
          complete_trajs.append(t)
      incomplete_trajs = temp_incomplete

    return complete_trajs if batched else complete_trajs[0]

  """
    Trajectories
  """
  def traj_fwd_logp(self, exp):
    """ Computes logp(trajectory) under current model.
        Batches over states in trajectory. 
    """
    states_to_logps = self.fwd_logps_unique(exp.traj[:-1])
    total = 0
    for state_to_logp, child in zip(states_to_logps, exp.traj[1:]):
      try:
        total += state_to_logp[child]
      except ValueError:
        # print(f'Hit ValueError. {child=}, {state_to_logp=}')
        print(f'Hit ValueError. child={child}, state_to_logp={state_to_logp}')
        import code; code.interact(local=dict(globals(), **locals()))
    return total

  def traj_back_logp(self, exp):
    """ Computes logp(trajectory) under current model.
        Batches over states in trajectory. 
    """
    states_to_logps = self.back_logps_unique(exp.traj[1:])
    total = 0
    for state_to_logp, parent in zip(states_to_logps, exp.traj[:-1]):
      total += state_to_logp[parent]
    return total

  def batch_traj_fwd_logp(self, batch):
    """ Computes logp(trajectory) under current model.
        Batches over all states in all trajectories in a batch.

        Batch: List of [trajectory]

        Returns: Tensor of batch_size, logp
    """
    trajs = [exp.traj for exp in batch]
    fwd_states, back_states, unroll_idxs = unroll_trajs(trajs)

    states_to_logps = self.fwd_logps_unique(fwd_states)

    fwd_logp_chosen = [s2lp[c] for s2lp, c in zip(states_to_logps, back_states)]
    
    fwd_chain = self.logZ.repeat(len(batch))
    fwd_chain = accumulate_by_traj(fwd_chain, fwd_logp_chosen, unroll_idxs)
    # fwd chain is [bsize]
    return fwd_chain
  
  def batch_traj_fwd_logp_unroll(self, batch):
    trajs = [exp.traj for exp in batch]
    fwd_states, back_states, unroll_idxs = unroll_trajs(trajs)

    states_to_logfs = self.fwd_logfs_unique(fwd_states)
    states_to_logps = self.fwd_logps_unique(fwd_states)
    fwd_logp_chosen = [s2lp[c] for s2lp, c in zip(states_to_logps, back_states)]

    log_F_s = torch.zeros((len(batch), self.mdp.forced_stop_len + 1)).to(device=self.args.device)
    log_pf_actions = torch.zeros((len(batch), self.mdp.forced_stop_len + 1)).to(device=self.args.device)
    for traj_idx, (start, end) in unroll_idxs.items():
      for i, j in enumerate(range(start, end)):
        log_F_s[traj_idx][i] = states_to_logfs[j]
        log_pf_actions[traj_idx][i] = fwd_logp_chosen[j]
        
    return log_F_s, log_pf_actions

  def batch_traj_back_logp(self, batch, evaluate = False):
    """ Computes logp(trajectory) under current model.
        Batches over all states in all trajectories in a batch.

        Batch: List of [trajectory]

        Returns: Tensor of batch_size, logp
    """
    trajs = [exp.traj for exp in batch]
    fwd_states, back_states, unroll_idxs = unroll_trajs(trajs)

    states_to_logps = self.back_logps_unique(back_states)
  
    if not evaluate:
      state_emb = self.policy_fwd.get_embedding(back_states)
      state_flow = self.flow_net(state_emb).squeeze()
      
      seq_len = int(len(state_flow)/self.args.num_samples_per_online_batch)
      state_flow[::seq_len] = 1

      with torch.no_grad():
        predict = self.rnd_predict.get_embedding(back_states)
        target = self.rnd_target.get_embedding(back_states)
        intrinsic_reward = torch.norm(predict - target, p=2, dim=-1).detach()
        intrinsic_reward *= self.args.reward_scale
        marginal_intrinsic_reward = intrinsic_reward[::seq_len]
        intrinsic_reward[::seq_len] = 0
    
      wandb.log({'intrinsic_reward': intrinsic_reward.mean().item()})

      back_logp_chosen = [s2lp[p]
      for s2lp, p, flow, int_reward in zip(states_to_logps, fwd_states, state_flow, intrinsic_reward)] 
      sum_reward = intrinsic_reward.view(-1, seq_len).sum(dim=1)
      back_chain = torch.stack([torch.log(torch.exp(exp.logr) + int_reward) for exp, int_reward in zip(batch, sum_reward)])
    else:
      back_logp_chosen = [s2lp[p] for s2lp, p in zip(states_to_logps, fwd_states)]
      back_chain = torch.stack([exp.logr for exp in batch])

    back_chain = accumulate_by_traj(back_chain, back_logp_chosen, unroll_idxs)
    # back_chain is [bsize]
    return back_chain
  



  def batch_traj_back_logp_unroll(self, batch):
    trajs = [exp.traj for exp in batch]
    fwd_states, back_states, unroll_idxs = unroll_trajs(trajs)

    states_to_logfs = self.back_logfs_unique(back_states)
    states_to_logps = self.back_logps_unique(back_states)
    back_logp_chosen = [s2lp[p] for s2lp, p in zip(states_to_logps, fwd_states)]
    
    log_F_next_s = torch.zeros((len(batch), self.mdp.forced_stop_len + 1)).to(device=self.args.device)
    log_pb_actions = torch.zeros((len(batch), self.mdp.forced_stop_len + 1)).to(device=self.args.device)
    for traj_idx, (start, end) in unroll_idxs.items():
      for i, j in enumerate(range(start, end)):
        log_F_next_s[traj_idx][i] = states_to_logfs[j]
        log_pb_actions[traj_idx][i] = back_logp_chosen[j]
        
    return log_F_next_s, log_pb_actions

  """
    Learning
  """
  def batch_loss_trajectory_balance(self, batch):
    """ batch: List of [Experience].

        Calls fwd_logps_unique and back_logps_unique (gpu) in parallel on
        all states in all trajs in batch, then collates.
    """
    fwd_chain = self.batch_traj_fwd_logp(batch)
    back_chain = self.batch_traj_back_logp(batch)

    # obtain target: mix back_chain with logp_guide
    targets = []
    for i, exp in enumerate(batch):
      if exp.logp_guide is not None:
        w = self.args.target_mix_backpolicy_weight
        log_rx = exp.logr.clone().detach().requires_grad_(True)
        target = w * back_chain[i] + (1 - w) * (exp.logp_guide + log_rx)
      else:
        target = back_chain[i]
      targets.append(target)
    targets = torch.stack(targets)

    losses = torch.square(fwd_chain - targets)
    # losses = torch.clamp(losses, max=5000)
    mean_loss = torch.mean(losses)
    return mean_loss


  def batch_loss_rnd(self, batch):
    trajs = [exp.traj for exp in batch]
    fwd_states, back_states, unroll_idxs = unroll_trajs(trajs)

    random_pi = self.rnd_target.get_embedding(fwd_states)
    predict_pi = self.rnd_predict.get_embedding(fwd_states)

    rnd_loss = torch.norm(predict_pi - random_pi, p=2, dim=-1).mean()
    # fwd chain is [bsize]
    return rnd_loss

  def train_tb(self, batch, log=True, online=False):
    """ Step on trajectory balance loss.

      Parameters
      ----------
      batch: List of [Experience]

      Batching is handled in trainers.py.
    """
    batch_loss = self.batch_loss_trajectory_balance(batch)
    rnd_loss = self.batch_loss_rnd(batch)
    batch_loss += rnd_loss
    for opt in self.optimizers:
      opt.zero_grad()
  
    batch_loss.backward()
    
    for param_set in self.clip_grad_norm_params:
      # torch.nn.utils.clip_grad_norm_(param_set, self.args.clip_grad_norm, error_if_nonfinite=True)
      torch.nn.utils.clip_grad_norm_(param_set, self.args.clip_grad_norm)
    for opt in self.optimizers:
      opt.step()
    self.clamp_logZ()

    if log:
      if online:
        self.online_loss_step += 1
        print(f'Online TB training:', batch_loss.item())
        wandb.log({'Online Regular TB loss': batch_loss.item(),
                   'online_loss_step': self.online_loss_step})
        wandb.log({'Online RND loss': rnd_loss.item(),
                   'online_loss_step': self.online_loss_step})
      else:
        self.offline_loss_step += 1
        print(f'Offline TB training:', batch_loss.item())
        wandb.log({'Offline Regular TB loss': batch_loss.item(),
                   'offline_loss_step': self.offline_loss_step})
        wandb.log({'Offline RND loss': rnd_loss.item(),
                   'offline_loss_step': self.offline_loss_step})
    return

  """ 
    IO & misc
  """
  def save_params(self, file):
    print('Saving checkpoint model ...')
    Path('/'.join(file.split('/')[:-1])).mkdir(parents=True, exist_ok=True)
    torch.save({
      'policy_fwd':   self.policy_fwd.state_dict(),
      'policy_back':  self.policy_back.state_dict(),
      'logZ':         self.logZ,
    }, file)
    return

  def load_for_eval_from_checkpoint(self, file):
    print(f'Loading checkpoint model ...')
    checkpoint = torch.load(file)
    self.policy_fwd.load_state_dict(checkpoint['policy_fwd'], cond=False)
    self.policy_back.load_state_dict(checkpoint['policy_back'], cond=False)
    self.logZ = checkpoint['logZ']
    for net in self.nets:
      net.eval()
    return

  def clip_policy_logits(self, scores):
    return torch.clip(scores, min=self.args.clip_policy_logit_min,
                              max=self.args.clip_policy_logit_max)


"""
  Trajectory/state rolling and accumulating
"""
def unroll_trajs(trajs):
  # Unroll trajectory into states: (num. trajs) -> (num. states)
  s1s, s2s = [], []
  traj_idx_to_batch_idxs = {}
  for traj_idx, traj in enumerate(trajs):
    start_idx = len(s1s)
    s1s += traj[:-1]
    s2s += traj[1:]
    end_idx = len(s1s)
    traj_idx_to_batch_idxs[traj_idx] = (start_idx, end_idx)
  return s1s, s2s, traj_idx_to_batch_idxs


def accumulate_by_traj(chain, batch_logp, traj_idx_to_batch_idxs):
  # Sum states by trajectory: (num. states) -> (num. trajs)
  for traj_idx, (start, end) in traj_idx_to_batch_idxs.items():
    chain[traj_idx] = chain[traj_idx] + sum(batch_logp[start:end])
  return chain
