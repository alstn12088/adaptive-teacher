"""
  seh as string
"""
import os
import pickle, functools
import numpy as np
from tqdm import tqdm
import torch

import gflownet.trainers as trainers
from gflownet.MDPs import molstrmdp
from gflownet.monitor import TargetRewardDistribution, Monitor
from gflownet.GFNs import models

from datasets.sehstr import gbr_proxy

from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.DataStructs import FingerprintSimilarity


class SEHstringMDP(molstrmdp.MolStrMDP):
  def __init__(self, args):
    super().__init__(args)
    self.args = args
    
    mode_info_file = args.mode_info_file
    assert args.blocks_file == 'datasets/sehstr/block_18.json', 'ERROR - x_to_r and rewards are designed for block_18.json'

    self.proxy_model = gbr_proxy.sEH_GBR_Proxy(args)
    
    with open('datasets/sehstr/block_18_stop6.pkl', 'rb') as f:
      self.oracle = pickle.load(f)

    with open('datasets/sehstr/sehstr_gbtr_allpreds.pkl', 'rb') as f:
      self.rewards = pickle.load(f)

    # scale rewards
    py = np.array(list(self.rewards))

    self.SCALE_REWARD_MIN = args.scale_reward_min
    self.SCALE_REWARD_MAX = args.scale_reward_max
    self.REWARD_EXP = args.beta
    self.REWARD_MAX = max(py)

    py = np.maximum(py, self.SCALE_REWARD_MIN)
    py = py ** self.REWARD_EXP
    self.scale = self.SCALE_REWARD_MAX / max(py)
    py = py * self.scale
    py = np.maximum(py, 1e-6)

    self.scaled_rewards = py
    self.scaled_oracle = {x: y for x, y in zip(self.oracle.keys(), py) if y > 0}
    assert min(self.scaled_oracle.values()) > 0

    # define modes as top % of xhashes and diversity metrics
    if args.mode_metric == "threshold":
      if os.path.exists(f"datasets/sehstr/modes_percentile_{args.mode_percentile}.pkl"):
        with open(f"datasets/sehstr/modes_percentile_{args.mode_percentile}.pkl", "rb") as f:
          self.modes = pickle.load(f)
      else:
        mode_percentile = args.mode_percentile
        self.mode_r_threshold = np.percentile(py, 100*(1-mode_percentile))
        num_modes = int(len(self.scaled_oracle) * mode_percentile)
        sorted_xs = sorted(self.scaled_oracle, key=self.scaled_oracle.get)
        self.modes = set(sorted_xs[-num_modes:])
        with open(f"datasets/sehstr/modes_percentile_{args.mode_percentile}.pkl", "wb") as f:
          pickle.dump(self.modes, f)
    elif args.mode_metric == "tanimoto":
      if os.path.exists(f"datasets/sehstr/modes_tanimoto_div{args.mode_div_threshold}_percentile_{args.mode_percentile}.pkl"):
        with open(f"datasets/sehstr/modes_tanimoto_div{args.mode_div_threshold}_percentile_{args.mode_percentile}.pkl", "rb") as f:
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
        with open(f"datasets/sehstr/modes_tanimoto_div{args.mode_div_threshold}_percentile_{args.mode_percentile}.pkl", "wb") as f:
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
    
    print(f"Beta: {self.REWARD_EXP}\tExpected reward: {self.expected_reward_scaled:.2f}\tLogZ: {self.logZ_scaled:.2f}")
    
    if os.path.exists(f"datasets/sehstr/samples_beta{args.beta}.pkl"):
      with open(f"datasets/sehstr/samples_beta{args.beta}.pkl", "rb") as f:
        true_samples_info = pickle.load(f)
      self.true_samples = true_samples_info["true_samples"]
      self.true_samples_scaled = true_samples_info["true_samples_scaled"]
    else:
      # generate samples from true distribution
      all_samples = list(self.oracle.keys())
      true_dist = np.exp(log_py * self.REWARD_EXP - self.logZ)
      true_samples_idx = np.random.choice(len(self.oracle), size=args.eval_num_samples, p=true_dist)
      self.true_samples = [self.state(all_samples[i], is_leaf=True) for i in true_samples_idx]

      true_dist_scaled = np.exp(log_py_scaled - self.logZ_scaled)
      true_samples_idx_scaled = np.random.choice(len(self.oracle), size=args.eval_num_samples, p=true_dist_scaled)
      self.true_samples_scaled = [self.state(all_samples[i], is_leaf=True) for i in true_samples_idx_scaled]
      
      with open(f"datasets/sehstr/samples_beta{args.beta}.pkl", "wb") as f:
        pickle.dump({"true_samples": self.true_samples, "true_samples_scaled": self.true_samples_scaled}, f)
    
    del self.oracle
    del self.scaled_oracle

  # Core
  @functools.lru_cache(maxsize=None)
  def reward(self, x):
    assert x.is_leaf, 'Error: Tried to compute reward on non-leaf node.'
    pred = self.proxy_model.predict_state(x)
    r = np.maximum(pred, self.SCALE_REWARD_MIN)
    r = r ** self.REWARD_EXP
    r = np.maximum(r * self.scale, 1e-32)
    return r
  
  def real_reward(self, x):
    assert x.is_leaf, 'Error: Tried to compute reward on non-leaf node.'
    return np.maximum(self.proxy_model.predict_state(x), self.SCALE_REWARD_MIN)

  def is_mode(self, x, r):
    if self.args.mode_metric == "threshold":
      return r >= self.mode_r_threshold
    else:
      return x.content in self.modes
  
  def unnormalize(self, r):
      r = r / self.scale
      r = r ** (1 / self.REWARD_EXP)
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
    target.init_from_base_rewards(self.scaled_rewards)
    return Monitor(self.args, target, dist_func=self.dist_states,
                   is_mode_f=self.is_mode,
                   unnormalize=self.unnormalize)

  def reduce_storage(self):
    del self.rewards
    del self.scaled_rewards


def main(args):
  print('Running experiment sehstr ...')
  if args.model == 'gafn':
    mdp = SEHstringMDP(args)
    actor = molstrmdp.MolStrActor(args, mdp)
    rnd_target = molstrmdp.MolStrActor(args, mdp)
    rnd_predict = molstrmdp.MolStrActor(args, mdp)
    model = models.make_model(args, mdp, actor, rnd_target = rnd_target, rnd_predict = rnd_predict)
    mdp.reduce_storage()
    trainer = trainers.Trainer(args, model, mdp)
  elif args.model == 'teacher':
    mdp_student = SEHstringMDP(args)
    mdp_teacher = SEHstringMDP(args)

    actor_student = molstrmdp.MolStrActor(args, mdp_student)
    actor_teacher = molstrmdp.MolStrActor(args, mdp_teacher)

    model_teacher, model_student = models.make_teacher_student_model(args, mdp_student, mdp_teacher, actor_student, actor_teacher)
    mdp_student.reduce_storage()
    mdp_teacher.reduce_storage()
    trainer = trainers.Trainer(args, model_student, mdp_student, teacher = model_teacher)
  else:
    mdp = SEHstringMDP(args)
    actor = molstrmdp.MolStrActor(args, mdp)
    model = models.make_model(args, mdp, actor)
    mdp.reduce_storage()
    trainer = trainers.Trainer(args, model, mdp)
  
  trainer.learn()
  return



def eval(args):
  print('Running evaluation sehstr ...')
  mdp = SEHstringMDP(args)
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
  # with torch.no_grad():
  #   eval_samples = model.batch_fwd_sample(args.eval_num_samples, epsilon=0.0)
    
  # allXtoR = dict()
  # for exp in total_samples:
  #   if exp.x not in allXtoR:
  #     allXtoR[exp.x] = exp.r 
  
  # results = trainer.evaluate(args.ckpt, eval_samples, allXtoR)
  all_modes = set()
  all_num_modes = list()
  for i, exp in enumerate(total_samples):
    if exp.x.content in mdp.modes:
      all_modes.add(exp.x.content)
    
    if i % (args.num_samples_per_online_batch * 100) == 0:
      all_num_modes.append(len(all_modes))
  results = {"all_num_modes": all_num_modes}
  print(results)
  results.update(vars(args))
  
  # round_num = 1
  # monitor.log_samples(round_num, eval_samples)
  # log = monitor.eval_samplelog(model, round_num, allXtoR)

  # # save results
  result_path = args.saved_models_dir + args.run_name
  # log_path = args.saved_models_dir + args.run_name
  postfix = ""
  if args.mode_metric == "threshold":
    postfix += f"_threshold_{args.mode_percentile}"
  elif args.mode_metric == "tanimoto":
    postfix += f"_tanimoto_div{args.mode_div_threshold}_percentile_{args.mode_percentile}"
  elif args.mode_metric == "hammingball":
    postfix += f"_hammingball_dist{args.mode_hammingball_dist}_percentile_{args.mode_percentile}"
  if args.ckpt == -1: # final
    result_path += '/' + f'final_results_{postfix}.pkl'
  else:
    result_path += '/' + f'round_{args.ckpt}_results_{postfix}.pkl'
    
  with open(result_path, "wb") as f:
    pickle.dump(results, f)
  return results
    
def number_of_modes(args):
  print('Running evaluation sehstr ...')
  mdp = SEHstringMDP(args)
  
  # load model checkpoint
  ckpt_path = args.saved_models_dir + args.run_name
  with open(ckpt_path + '/' + f"final_sample.pkl", "rb") as f:
    generated_samples = pickle.load(f)
    
  unique_modes = set()
  batch_size = args.num_samples_per_online_batch
  number_of_modes = np.zeros((len(generated_samples) // batch_size, ))
  with tqdm(total=len(generated_samples)) as pbar:
    for i in range(0, len(generated_samples), batch_size):
      for exp in generated_samples[i: i+batch_size]:
        if mdp.is_mode(exp.x, exp.r) and exp.x.content not in unique_modes:
          unique_modes.add(exp.x.content)
          number_of_modes[i // batch_size] += 1
      pbar.update(batch_size)
      pbar.set_postfix(number_of_modes=np.sum(number_of_modes))
  print(np.sum(number_of_modes))
  np.savez_compressed(ckpt_path + '/' + f'number_of_modes.npz', modes=number_of_modes) 

