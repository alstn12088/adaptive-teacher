"""
  qm9 as string
"""
import os
import pickle, functools
import numpy as np
from tqdm import tqdm
import torch

import gflownet.trainers as trainers
from gflownet.MDPs import molstrmdp
from gflownet.monitor import TargetRewardDistribution, Monitor, diversity
from gflownet.GFNs import models

from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.DataStructs import FingerprintSimilarity


class QM9stringMDP(molstrmdp.MolStrMDP):
  def __init__(self, args):
    super().__init__(args)
    self.args = args

    x_to_r_file = args.x_to_r_file
    # mode_info_file = args.mode_info_file

    # Read from file
    print(f'Loading data ...')
    with open(x_to_r_file, 'rb') as f:
      self.oracle = pickle.load(f)
    
    # scale rewards
    py = np.array(list(self.oracle.values()))

    self.SCALE_REWARD_MIN = args.scale_reward_min
    self.SCALE_REWARD_MAX = args.scale_reward_max
    self.REWARD_EXP = args.beta
    self.REWARD_MAX = max(py)

    py = np.maximum(py, self.SCALE_REWARD_MIN)
    py = py ** self.REWARD_EXP
    self.scale = self.SCALE_REWARD_MAX / max(py)
    py = py * self.scale + 1e-12

    self.scaled_oracle = {x: y for x, y in zip(self.oracle.keys(), py) if y > 0}
    assert min(self.scaled_oracle.values()) > 0

    # define modes as top % of xhashes and diversity metrics
    if args.mode_metric == "threshold":
      if os.path.exists(f"datasets/qm9str/modes_percentile_{args.mode_percentile}.pkl"):
        with open(f"datasets/qm9str/modes_percentile_{args.mode_percentile}.pkl", "rb") as f:
          self.modes = pickle.load(f)
      else:
        mode_percentile = args.mode_percentile
        self.mode_r_threshold = np.percentile(py, 100*(1-mode_percentile))
        num_modes = int(len(self.scaled_oracle) * mode_percentile)
        sorted_xs = sorted(self.scaled_oracle, key=self.scaled_oracle.get)
        self.modes = set(sorted_xs[-num_modes:])
        with open(f"datasets/qm9str/modes_percentile_{args.mode_percentile}.pkl", "wb") as f:
          pickle.dump(self.modes, f)
    elif args.mode_metric == "tanimoto":
      if os.path.exists(f"datasets/qm9str/modes_tanimoto_div{args.mode_div_threshold}_percentile_{args.mode_percentile}.pkl"):
        with open(f"datasets/qm9str/modes_tanimoto_div{args.mode_div_threshold}_percentile_{args.mode_percentile}.pkl", "rb") as f:
          self.modes = pickle.load(f)
      else:
        mode_percentile = args.mode_percentile
        self.mode_r_threshold = np.percentile(py, 100*(1-mode_percentile))
        self.mode_div_threshold = args.mode_div_threshold
        
        print(f"Computing modes using tanimoto similarity with threshold {self.mode_div_threshold}...")
        self.modes = set()
        for x, y in tqdm(self.scaled_oracle.items()):
          y = self.scaled_oracle[x]
          if y >= self.mode_r_threshold:
            if len(self.modes) == 0:
              self.modes.add(x)
            else:
              flag = False
              for mode in self.modes:
                diversity_score = self.dist_states(x, mode)
                if diversity_score <= self.mode_div_threshold:
                  flag = True
                  break
              if not flag:
                self.modes.add(x)   
        with open(f"datasets/qm9str/modes_tanimoto_div{args.mode_div_threshold}_percentile_{args.mode_percentile}.pkl", "wb") as f:
          pickle.dump(self.modes, f)
        
    print(f"Mode metric: {args.mode_metric}\tFound num modes: {len(self.modes)}")
    
    # compute true expected reward and logz
    py = np.array(list(self.oracle.values()))
    py = np.maximum(py, self.SCALE_REWARD_MIN)
    log_py = np.log(py)
    
    self.logZ = torch.logsumexp(torch.from_numpy(log_py * self.REWARD_EXP), 0).item()
    logr_square = torch.logsumexp(torch.from_numpy(log_py * (self.REWARD_EXP + 1)), 0).item()
    self.expected_reward = np.exp(logr_square - self.logZ)
    
    log_py_scaled = log_py * self.REWARD_EXP
    log_scale = np.log(self.SCALE_REWARD_MAX) - max(log_py_scaled)
    log_py_scaled = log_py_scaled + log_scale
    self.logZ_scaled = torch.logsumexp(torch.from_numpy(log_py_scaled).float(), dim=0).item()
    logr_square_scaled = torch.logsumexp(torch.from_numpy(log_py_scaled + log_py).float(), dim=0).item()
    self.expected_reward_scaled = np.exp(logr_square_scaled - self.logZ_scaled)
    
    print(f"Beta: {self.REWARD_EXP}\tExpected reward: {self.expected_reward:.2f}\tLogZ: {self.logZ:.2f}")
    
    # generate samples from true distribution
    all_samples = list(self.oracle.keys())
    true_dist = np.exp(log_py * self.REWARD_EXP - self.logZ)
    true_samples_idx = np.random.choice(len(self.oracle), size=args.eval_num_samples, p=true_dist)
    self.true_samples = [self.state(all_samples[i], is_leaf=True) for i in true_samples_idx]

    true_dist_scaled = np.exp(log_py_scaled - self.logZ_scaled)
    true_samples_idx_scaled = np.random.choice(len(self.oracle), size=args.eval_num_samples, p=true_dist_scaled)
    self.true_samples_scaled = [self.state(all_samples[i], is_leaf=True) for i in true_samples_idx_scaled]


  # Core
  def reward(self, x):
    assert x.is_leaf, 'Error: Tried to compute reward on non-leaf node.'
    return self.scaled_oracle[x.content]
  
  def real_reward(self, x):
    assert x.is_leaf, 'Error: Tried to compute reward on non-leaf node.'
    return np.maximum(self.oracle[x.content], self.SCALE_REWARD_MIN)

  def is_mode(self, x, r):
    return x.content in self.modes
  
  def unnormalize(self, r):
      r = r / self.scale
      r = r ** (1 / self.REWARD_EXP)
      r = r / self.REWARD_MAX
      return r

  # Diversity
  def dist_states(self, state1, state2):
    """ Tanimoto similarity on morgan fingerprints """
    fp1 = self.get_morgan_fp(state1)
    fp2 = self.get_morgan_fp(state2)
    return 1 - FingerprintSimilarity(fp1, fp2)

  @functools.lru_cache(maxsize=None)
  def get_morgan_fp(self, state):
    mol = self.state_to_mol(state)
    fp = GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    return fp

  """
    Interpretation & visualization
  """
  def make_monitor(self):
    """ Make monitor, called during training. """
    target = TargetRewardDistribution()
    rs_all = list(self.scaled_oracle.values())
    target.init_from_base_rewards(rs_all)
    return Monitor(self.args, target, dist_func=self.dist_states,
                   is_mode_f=self.is_mode,
                   unnormalize=self.unnormalize)


def main(args):
  print('Running experiment qm9str ...')
 
  if args.model == 'gafn':
    mdp = QM9stringMDP(args)
    actor = molstrmdp.MolStrActor(args, mdp)
    rnd_target = molstrmdp.MolStrActor(args, mdp)
    rnd_predict = molstrmdp.MolStrActor(args, mdp)
    model = models.make_model(args, mdp, actor, rnd_target = rnd_target, rnd_predict = rnd_predict)
    trainer = trainers.Trainer(args, model, mdp)
  elif args.model == 'teacher':
    mdp_student = QM9stringMDP(args)
    mdp_teacher = QM9stringMDP(args)

    actor_student = molstrmdp.MolStrActor(args, mdp_student)
    actor_teacher = molstrmdp.MolStrActor(args, mdp_teacher)

    model_teacher, model_student = models.make_teacher_student_model(args, mdp_student, mdp_teacher, actor_student, actor_teacher)

    trainer = trainers.Trainer(args, model_student, mdp_student, teacher = model_teacher)
  else:
    mdp = QM9stringMDP(args)
    actor = molstrmdp.MolStrActor(args, mdp)
    model = models.make_model(args, mdp, actor)
    # monitor = mdp.make_monitor()
    trainer = trainers.Trainer(args, model, mdp)
  trainer.learn()
  return

def eval(args):
  print('Running evaluation qm9str ...')
  mdp = QM9stringMDP(args)
  actor = molstrmdp.MolStrActor(args, mdp)
  model = models.make_model(args, mdp, actor)
  # monitor = mdp.make_monitor()
  trainer = trainers.Trainer(args, model, mdp, actor)
  
  # load model checkpoint
  ckpt_path = args.saved_models_dir + args.run_name
  if args.ckpt == -1: # final
    model.load_for_eval_from_checkpoint(ckpt_path + '/' + 'final.pth')
    with open(ckpt_path + '/' + f"final_sample.pkl", "rb") as f:
      total_samples = pickle.load(f)
  else:
    model.load_for_eval_from_checkpoint(ckpt_path + '/' + f'round_{args.ckpt}.pth')
    with open(ckpt_path + '/' + f"round_{args.ckpt}_sample.pkl", "rb") as f:
      total_samples = pickle.load(f)
    
  # evaluate
  with torch.no_grad():
    eval_samples = model.batch_fwd_sample(args.eval_num_samples, epsilon=0.0)
    
  allXtoR = dict()
  for exp in total_samples:
    if exp.x not in allXtoR:
      allXtoR[exp.x] = exp.r 
  
  results = trainer.evaluate(args.ckpt, eval_samples, allXtoR)
  results.update(vars(args))
  
  # round_num = 1
  # monitor.log_samples(round_num, eval_samples)
  # log = monitor.eval_samplelog(model, round_num, allXtoR)

  # # save results
  result_path = args.saved_models_dir + args.run_name
  # log_path = args.saved_models_dir + args.run_name
  if args.ckpt == -1: # final
    result_path += '/' + 'final_results.pkl'
  else:
    result_path += '/' + f'round_{args.ckpt}_results.pkl'
    
  with open(result_path, "wb") as f:
    pickle.dump(results, f)
  return results
    
def number_of_modes(args):
  print('Count number of modes qm9str ...')
  
  # load model checkpoint
  ckpt_path = args.saved_models_dir + args.run_name
  with open(ckpt_path + '/' + f"final_sample.pkl", "rb") as f:
    generated_samples = pickle.load(f)
    
  with open(args.mode_info_file, "rb") as f:
    mode_info = pickle.load(f)
    
  unique_samples = set()
  batch_size = args.num_samples_per_online_batch
  number_of_modes = {k: np.zeros((len(generated_samples) // batch_size, )) for k in mode_info}
  with tqdm(total=len(generated_samples)) as pbar:
    for i in range(0, len(generated_samples), batch_size):
      for exp in generated_samples[i: i+batch_size]:
        if exp.x not in unique_samples:
          if exp.x.content in mode_info["modes_div_threshold_075"]:
            number_of_modes["modes_div_threshold_075"][i // batch_size] += 1
          if exp.x.content in mode_info["modes_div_threshold_05"]:
            number_of_modes["modes_div_threshold_05"][i // batch_size] += 1
          if exp.x.content in mode_info["modes"]:
            number_of_modes["modes"][i // batch_size] += 1
        unique_samples.add(exp.x)
      pbar.update(batch_size)
      pbar.set_postfix(number_of_modes=np.sum(number_of_modes["modes"]))
  print(np.sum(number_of_modes["modes"]))
  np.savez_compressed(ckpt_path + '/' + f'number_of_modes_updated.npz', modes=number_of_modes["modes"],
                                                                        modes_div_threshold_05=number_of_modes["modes_div_threshold_05"],
                                                                        modes_div_threshold_075=number_of_modes["modes_div_threshold_075"],) 
