import math
import random, time
import pickle
import numpy as np
import torch
import wandb
from scipy import stats
from tqdm import tqdm
from collections import deque, OrderedDict
# import ray

from . import guide
from .data import Experience

class FixSizeOrderedDict(OrderedDict):
    def __init__(self, *args, max=0, **kwargs):
        self._max = max
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        OrderedDict.__setitem__(self, key, value)
        if self._max > 0:
            if len(self) > self._max:
                self.popitem(False)

class Trainer:
  def __init__(self, args, model, mdp, teacher=None):
    self.args = args
    self.model = model
    self.mdp = mdp
    self.teacher = teacher

  def learn(self, *args, **kwargs):
    print(f'Learning without guide workers ...')
    self.learn_default(*args, **kwargs)

  def handle_init_dataset(self, initial_XtoR):
    if initial_XtoR:
      print(f'Using initial dataset of size {len(initial_XtoR)}. \
              Skipping first online round ...')
      if self.args.init_logz:
        self.model.init_logz(np.log(sum(initial_XtoR.values())))
    else:
      print(f'No initial dataset used')
    return

  """
    Training
  """
  def learn_default(self, initial_XtoR=None):
    """ Main learning training loop.
        Each learning round:
          Each online batch:
            sample a new dataset using exploration policy.
          Each offline batch:
            resample batch from full historical dataset
        Monitor exploration - judge modes with monitor_explore callable.

        To learn on fixed dataset only: Set 0 online batches per round,
        and provide initial dataset.

        dataset = List of [Experience]
    """
    allXtoR = initial_XtoR if initial_XtoR else dict()
    allXtoL = dict()
    allXtoR_buffer = FixSizeOrderedDict(max=self.args.replay_buffer_size)
    allXtoL_buffer = FixSizeOrderedDict(max=self.args.replay_buffer_size)
    self.handle_init_dataset(initial_XtoR)

    num_online = self.args.num_online_batches_per_round
    num_offline = self.args.num_offline_batches_per_round
    online_bsize = self.args.num_samples_per_online_batch
    offline_bsize = self.args.num_samples_per_offline_batch
    
    eval_num_samples = self.args.eval_num_samples
    # monitor_fast_every = self.args.monitor_fast_every
    # monitor_num_samples = self.args.monitor_num_samples
    # print(f'Starting active learning. \
    #         Each round: {num_online=}, {num_offline=}')
    print(f'Starting active learning. \
            Each round: num_online={num_online}, num_offline={num_offline}')
    total_samples = []
    accepted_samples = []
    for round_num in tqdm(range(self.args.num_active_learning_rounds)):
      print(f'Starting learning round {round_num+1} / {self.args.num_active_learning_rounds} ...')
      
      # Online training - skip first if initial dataset was provided
      if not initial_XtoR or round_num > 0:
        for _ in range(num_online):
          if self.args.ls:
            if self.args.deterministic:
              with torch.no_grad():
                explore_data = self.model.batch_fwd_sample_ls(online_bsize,
                  epsilon=self.args.explore_epsilon, k=self.args.k, i=self.args.i, deterministic=True)
            else:
              with torch.no_grad():
                explore_data = self.model.batch_fwd_sample_ls(online_bsize,
                  epsilon=self.args.explore_epsilon, k=self.args.k, i=self.args.i, deterministic=False)
          elif self.args.model == "mars":
            with torch.no_grad():
              if len(total_samples) == 0:
                explore_data = self.model.batch_fwd_sample(online_bsize,
                    epsilon=self.args.explore_epsilon)
              else:
                explore_data, accepted_data = self.model.batch_fwd_sample(online_bsize,
                    epsilon=self.args.explore_epsilon, explore_data=explore_data) 
                accepted_samples.extend(accepted_data)
          else:
            if self.args.model == 'teacher':
              with torch.no_grad():
                if round_num % 3 == 0 or round_num % 3 == 1:
                  explore_data = self.model.batch_fwd_sample(online_bsize,
                      epsilon=self.args.explore_epsilon)
                else:
                  explore_data = self.teacher.batch_fwd_sample(online_bsize,
                      epsilon=self.args.explore_epsilon)
                # num_student_batch = int(online_bsize * self.args.teacher_ratio)
                # num_teacher_batch = online_bsize - num_student_batch
                # explore_data_student = self.model.batch_fwd_sample(num_student_batch,
                #     epsilon=self.args.explore_epsilon)
                # explore_data_teacher = self.teacher.batch_fwd_sample(num_teacher_batch,
                #     epsilon=self.args.explore_epsilon)
                # explore_data = explore_data_student + explore_data_teacher
            
            else:           
              with torch.no_grad():
                explore_data = self.model.batch_fwd_sample(online_bsize,
                    epsilon=self.args.explore_epsilon)


          
          # Train on online dataset
          if self.args.model in ["a2c", "sql", "mars"]:
            pass
          elif self.args.model == "ppo":
            # As ppo is on-policy algorithm, we double online training steps
            for step_num in range(self.args.num_steps_per_batch * 2):
              self.model.train(explore_data)   
          else:
            for step_num in range(self.args.num_steps_per_batch):
              adv_reward = self.model.train(explore_data)
              if self.args.model == 'teacher':
                self.teacher.train(explore_data, torch.log(adv_reward).detach(), round_num)


          # Save to full dataset
          
          #if not self.args.model == 'teacher' or True:
          for i, exp in enumerate(explore_data):
            if exp.x not in allXtoR:
              allXtoR[exp.x] = exp.r
              allXtoR_buffer[exp.x] = exp.r
            if self.args.per:
              allXtoL[exp.x] = adv_reward[i].item()    
              allXtoL_buffer[exp.x] = adv_reward[i].item()  
          # else:
          #   for i, exp in enumerate(explore_data_teacher):
          #     if exp.x not in allXtoR:
          #       allXtoR[exp.x] = exp.r
          #       allXtoR_buffer[exp.x] = exp.r
          #     if self.args.per:
          #       allXtoL[exp.x] = adv_reward[i].item()    
          #       allXtoL_buffer[exp.x] = adv_reward[i].item()

          total_samples.extend(explore_data)

      # Offline training
      for _ in range(num_offline):
        if self.args.model == "a2c" or self.args.model == "sql":
          # we do not use PRT for RL-based methods
          # As A2C and SQL are off-policy algorithm, we double offline training steps
          offline_dataset = random.choices(total_samples, k=offline_bsize)
          for step_num in range(self.args.offline_num_steps_per_batch * 2):
            self.model.train(offline_dataset)
        elif self.args.model == "ppo":
          pass
        elif self.args.model == "mars":
          if len(accepted_samples) >= offline_bsize:
            offline_dataset = random.choices(accepted_samples, k=offline_bsize)
            for step_num in range(self.args.offline_num_steps_per_batch):
              self.model.train(offline_dataset)
        else:
          # offline_xs = self.select_offline_xs(allXtoR, offline_bsize)
          if self.args.per:
            offline_xs = self.select_offline_xs_per(allXtoR_buffer, allXtoL_buffer, offline_bsize)
            offline_dataset = self.offline_PB_traj_sample(offline_xs, allXtoR)
          else:
            offline_xs = self.select_offline_xs(allXtoR_buffer, offline_bsize)
            offline_dataset = self.offline_PB_traj_sample(offline_xs, allXtoR)
          for step_num in range(self.args.offline_num_steps_per_batch):
            adv_reward = self.model.train(offline_dataset)
            if self.args.model == 'teacher':
              self.teacher.train(offline_dataset, torch.log(adv_reward).detach(), round_num)

        if self.args.per:
          for i, exp in enumerate(offline_dataset):          
              allXtoL[exp.x] = adv_reward[i].item()    
              allXtoL_buffer[exp.x] = adv_reward[i].item()  


      
      if round_num and round_num % self.args.eval_every_x_active_rounds == 0:
        print(f'Evaluating round {round_num} ...')
        start_time = time.time()
        onpolicy_samples = self.model.batch_fwd_sample(eval_num_samples, epsilon=0)
        results = self.evaluate(round_num, onpolicy_samples, allXtoR)
        wandb.log(results)
        print(f'Completed\t{time.time() - start_time:.2f}s')

      if round_num and round_num % self.args.save_every_x_active_rounds == 0:
        self.model.save_params(self.args.saved_models_dir + \
                               self.args.run_name + "/" + f'round_{round_num}.pth')
        with open(self.args.saved_models_dir + \
                  self.args.run_name + "/" + f"round_{round_num}_sample.pkl", "wb") as f:
          pickle.dump(total_samples, f)

    print(f'Evaluating round {round_num} ...')
    start_time = time.time()
    onpolicy_samples = self.model.batch_fwd_sample(eval_num_samples, epsilon=0)
    results = self.evaluate(round_num, onpolicy_samples, allXtoR)
    wandb.log(results)
    print(f'Completed\t{time.time() - start_time:.2f}s')


    print('Finished training.')
    self.model.save_params(self.args.saved_models_dir + \
                           self.args.run_name + "/" + 'final.pth')
    self.model.save_params(self.args.saved_models_dir + \
                           self.args.run_name + "/" + f'round_{round_num+1}.pth')
    with open(self.args.saved_models_dir + \
          self.args.run_name + "/" + f"final_sample.pkl", "wb") as f:
      pickle.dump(total_samples, f)
    with open(self.args.saved_models_dir + \
          self.args.run_name + "/" + f"round_{round_num+1}_sample.pkl", "wb") as f:
      pickle.dump(total_samples, f)
    # self.monitor.maybe_eval_samplelog(self.model, round_num, allXtoR)
    return


  """
    Offline training
  """
  def select_offline_xs(self, allXtoR, batch_size):
    select = self.args.get('offline_select', 'prt')
    if select == 'prt':
      return self.__biased_sample_xs(allXtoR, batch_size)
    elif select == 'random':
      return self.__random_sample_xs(allXtoR, batch_size)

  def select_offline_xs_per(self, allXtoR, allXtoL, batch_size):
    select = self.args.get('offline_select', 'prt')
    if select == 'prt':
      return self.__biased_sample_xs_per(allXtoR, allXtoL, batch_size)
    elif select == 'random':
      return self.__random_sample_xs(allXtoR, batch_size)


  def __biased_sample_xs(self, allXtoR, batch_size):
    """ Select xs for offline training. Returns List of [State].
        Draws 50% from top 10% of rewards, and 50% from bottom 90%. 
    """
    if len(allXtoR) < 10:
      return []
    rewards = np.array(list(allXtoR.values()))
    threshold = np.percentile(rewards, 90)
    top_xs = [x for x, r in allXtoR.items() if r >= threshold]
    bottom_xs = [x for x, r in allXtoR.items() if r <= threshold]
    sampled_xs = random.choices(top_xs, k=batch_size // 2) + \
                 random.choices(bottom_xs, k=batch_size // 2)
    return sampled_xs


  def __biased_sample_xs_per(self, allXtoR, allXtoL, batch_size):
    """ Select xs for offline training. Returns List of [State].
        Draws 50% from top 10% of rewards, and 50% from bottom 90%. 
    """
    if len(allXtoR) < 10:
      return []
    rewards = np.array(list(allXtoL.values()))
    threshold = np.percentile(rewards, 90)
    top_xs = [x for x, r in allXtoL.items() if r >= threshold]
    bottom_xs = [x for x, r in allXtoL.items() if r <= threshold]

    sampled_xs = random.choices(top_xs, k=batch_size // 2) + \
                 random.choices(bottom_xs, k=batch_size // 2)
    return sampled_xs

  def __random_sample_xs(self, allXtoR, batch_size):
    """ Select xs for offline training. Returns List of [State]. """
    return random.choices(list(allXtoR.keys()), k=batch_size)

  def offline_PB_traj_sample(self, offline_xs, allXtoR):
    """ Sample trajectories for x using P_B, for offline training with TB.
        Returns List of [Experience].
    """
    if allXtoR is None:
      offline_rs = [self.mdp.reward(x) for x in offline_xs]
    else:
      offline_rs = [allXtoR[x] for x in offline_xs]

    # Not subgfn: sample trajectories from backward policy
    print(f'Sampling trajectories from backward policy ...')
    with torch.no_grad():
      offline_trajs = self.model.batch_back_sample(offline_xs)

    offline_dataset = [
      Experience(traj=traj, x=x, r=r,
                  logr=torch.log(torch.tensor(r, dtype=torch.float32,device=self.args.device)))
      for traj, x, r in zip(offline_trajs, offline_xs, offline_rs)
    ]
    return offline_dataset


  def evaluate(self, round_num, samples, allXtoR):
    real_rewards = []
    for exp in samples:
      real_rewards.append(self.mdp.real_reward(exp.x))  
    real_rewards = np.array(real_rewards)
    real_rewards = np.maximum(real_rewards, self.args.scale_reward_min)
    
    # Metric 1. Gap Between True Expected Reward and Estimated Expected Reward
    estimated_expected_reward = np.mean(real_rewards)
    estimated_expected_reward_gap = np.abs(estimated_expected_reward - self.mdp.expected_reward)
    estimated_expected_reward_gap_scaled = np.abs(estimated_expected_reward - self.mdp.expected_reward_scaled)
    
    # Metric 2. Gap Between True LogZ and Estimated LogZ / elbo and eubo
    fwd_chain = self.model.batch_traj_fwd_logp(samples) - self.model.logZ.item()
    
    if self.args.model == 'gafn':
      back_chain = self.model.batch_traj_back_logp(samples, evaluate=True)
    else:
      back_chain = self.model.batch_traj_back_logp(samples)
    estimated_logZ = torch.logsumexp(back_chain - fwd_chain, 0).item() - math.log(len(samples))
    estimated_logZ_gap = np.abs(estimated_logZ - self.mdp.logZ)
    estimated_logZ_gap_scaled = np.abs(estimated_logZ - self.mdp.logZ_scaled)
    
    elbo = (back_chain - fwd_chain).mean().item()
    
    eubo = []
    eubo_scaled = []
    bsz = 128
    for i in range(0, len(self.mdp.true_samples), bsz):
      true_samples = self.offline_PB_traj_sample(self.mdp.true_samples[i:i+bsz], None)
      fwd_chain_true = self.model.batch_traj_fwd_logp(true_samples).cpu().detach().numpy() - self.model.logZ.item()
      
      if self.args.model == 'gafn':
        back_chain_true = self.model.batch_traj_back_logp(true_samples, evaluate=True).cpu().detach().numpy()
      else:
        back_chain_true = self.model.batch_traj_back_logp(true_samples).cpu().detach().numpy()
      
      eubo.append((back_chain_true - fwd_chain_true))
      
      true_samples_scaled = self.offline_PB_traj_sample(self.mdp.true_samples_scaled[i:i+bsz], None)
      fwd_chain_true_scaled = self.model.batch_traj_fwd_logp(true_samples_scaled).cpu().detach().numpy() - self.model.logZ.item()
      
      if self.args.model == 'gafn':
        back_chain_true_scaled = self.model.batch_traj_back_logp(true_samples_scaled, evaluate=True).cpu().detach().numpy()
      else:
        back_chain_true_scaled = self.model.batch_traj_back_logp(true_samples_scaled).cpu().detach().numpy()
      
      eubo_scaled.append((back_chain_true_scaled - fwd_chain_true_scaled))
    # true_samples = self.offline_PB_traj_sample(self.mdp.true_samples, None)
    # fwd_chain_true = self.model.batch_traj_fwd_logp(true_samples).cpu().detach().numpy() - self.model.logZ.item()
    
    # if self.args.model == 'gafn':
    #   back_chain_true = self.model.batch_traj_back_logp(true_samples, evaluate=True).cpu().detach().numpy()
    # else:
    #   back_chain_true = self.model.batch_traj_back_logp(true_samples).cpu().detach().numpy()
    
    # eubo = (back_chain_true - fwd_chain_true).mean().item()
    
    # true_samples_scaled = self.offline_PB_traj_sample(self.mdp.true_samples_scaled, None)
    # fwd_chain_true_scaled = self.model.batch_traj_fwd_logp(true_samples_scaled).cpu().detach().numpy() - self.model.logZ.item()
    
    # if self.args.model == 'gafn':
    #   back_chain_true_scaled = self.model.batch_traj_back_logp(true_samples_scaled, evaluate=True).cpu().detach().numpy()
    # else:
    #   back_chain_true_scaled = self.model.batch_traj_back_logp(true_samples_scaled).cpu().detach().numpy()
    
    # eubo_scaled = (back_chain_true_scaled - fwd_chain_true_scaled).mean().item()
    eubo = np.concatenate(eubo).mean().item()
    eubo_scaled = np.concatenate(eubo_scaled).mean().item()
    
    # Metric 3. Pearson Correlation Coefficient
    pearson_corr = stats.pearsonr(fwd_chain_true, 
                                  np.log(np.array([self.mdp.real_reward(exp.x) for exp in true_samples])))[0]
    pearson_corr_scaled = stats.pearsonr(fwd_chain_true_scaled, 
                                         np.array([exp.logr.cpu().detach().numpy() for exp in true_samples_scaled]))[0]
    
    # Metric 4. Number of modes
    onpolicy_xs = set([exp.x for exp in samples])
    onpolicy_num_modes = 0
    for x in onpolicy_xs:
      if x.content in self.mdp.modes:
        onpolicy_num_modes += 1
    
    all_num_modes = 0
    for x, _ in allXtoR.items():
      if x.content in self.mdp.modes:
        all_num_modes += 1
    
    results = {
      "round_num": round_num,
      "estimated_expected_reward": estimated_expected_reward,
      "estimated_expected_reward_gap": estimated_expected_reward_gap,
      "estimated_expected_reward_gap_scaled": estimated_expected_reward_gap_scaled,
      "estimated_logZ": estimated_logZ,
      "estimated_logZ_gap": estimated_logZ_gap,
      "estimated_logZ_gap_scaled": estimated_logZ_gap_scaled,
      "elbo": elbo,
      "eubo": eubo,
      "eubo_scaled": eubo_scaled,
      "pearson_corr": pearson_corr,
      "pearson_corr_scaled": pearson_corr_scaled,
      "onpolicy_num_modes": onpolicy_num_modes,
      "all_num_modes": all_num_modes,
    }

    print(results)
    return results