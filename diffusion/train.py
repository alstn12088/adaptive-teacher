from plot_utils import *
import argparse
import torch
import os

from utils import set_seed, cal_subtb_coef_matrix, fig_to_image, get_gfn_optimizer, get_gfn_forward_loss, \
    get_gfn_backward_loss, get_exploration_std, get_name
from buffer import ReplayBuffer
from models import GFN
from gflownet_losses import *
from energies import *
from evaluations import *
import matplotlib.pyplot as plt
from tqdm import trange
from torch.optim.lr_scheduler import MultiStepLR
from langevin import langevin_dynamics
import copy



WANDB = True

if WANDB:
    import wandb

parser = argparse.ArgumentParser(description='GFN Linear Regression')
parser.add_argument('--lr_policy', type=float, default=1e-3)
parser.add_argument('--lr_flow', type=float, default=1e-1)
parser.add_argument('--lr_back', type=float, default=1e-3)
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--s_emb_dim', type=int, default=256)
parser.add_argument('--t_emb_dim', type=int, default=256)
parser.add_argument('--harmonics_dim', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--epochs', type=int, default=10000)
parser.add_argument('--buffer_size', type=int, default=50000)
parser.add_argument('--T', type=int, default=100)
parser.add_argument('--subtb_lambda', type=int, default=2)
parser.add_argument('--t_scale', type=float, default=1.)
parser.add_argument('--log_var_range', type=float, default=4.)
parser.add_argument('--energy', type=str, default='25gmm',
                    choices=('25gmm', 'many_well'))
parser.add_argument('--mode_fwd', type=str, default="tb")
parser.add_argument('--mode_bwd', type=str, default="tb")
parser.add_argument('--both_ways', action='store_true', default=False)
parser.add_argument('--per', action='store_true', default=False)


# For back-and-forth local search
################################################################
parser.add_argument('--ls', action='store_true', default=False)
parser.add_argument('--els', action='store_true', default=False)
parser.add_argument('--target_acceptance_rate', type=float, default=0.574)
parser.add_argument('--max_iter_ls', type=int, default=200)
parser.add_argument('--burn_in', type=int, default=100)
# How frequently to make local search
parser.add_argument('--ls_cycle', type=int, default=100)

# langevin step size
parser.add_argument('--ld_step', type=float, default=0.001)

parser.add_argument('--ld_schedule', action='store_true', default=True)

# how many steps to take in the local search
parser.add_argument('--mcmc_steps', type=int, default=100)

parser.add_argument('--mcmc_K', type=int, default=50)

parser.add_argument('--ls_freq', type=int, default=10)
parser.add_argument('--adv_freq', type=int, default=2)
parser.add_argument('--reward_filtering', action='store_true', default=False)

# For replay buffer
################################################################
# high beta give steep priorization in reward prioritized replay sampling
parser.add_argument('--beta', type=float, default=1.)

# low rank_weighted give steep priorization in rank-based replay sampling
parser.add_argument('--rank_weight', type=float, default=1e-2)

# three kinds of replay training: random, reward prioritized, rank-based
parser.add_argument('--prioritized', type=str, default="rank", choices=('none', 'reward', 'rank'))
################################################################

parser.add_argument('--bwd', action='store_true', default=False)
parser.add_argument('--exploratory', action='store_true', default=False)

parser.add_argument('--langevin', action='store_true', default=False)
parser.add_argument('--langevin_scaling_per_dimension', action='store_true', default=False)
parser.add_argument('--conditional_flow_model', action='store_true', default=False)
parser.add_argument('--learn_pb', action='store_true', default=False)
parser.add_argument('--pb_scale_range', type=float, default=0.1)
parser.add_argument('--learned_variance', action='store_true', default=False)
parser.add_argument('--partial_energy', action='store_true', default=False)
parser.add_argument('--exploration_factor', type=float, default=0.1)
parser.add_argument('--exploration_wd', action='store_true', default=False)
parser.add_argument('--clipping', action='store_true', default=False)
parser.add_argument('--lgv_clip', type=float, default=1e2)
parser.add_argument('--gfn_clip', type=float, default=1e4)
parser.add_argument('--zero_init', action='store_true', default=True)
parser.add_argument('--pis_architectures', action='store_true', default=False)
parser.add_argument('--lgv_layers', type=int, default=3)
parser.add_argument('--joint_layers', type=int, default=2)
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--weight_decay', type=float, default=1e-7)
parser.add_argument('--use_weight_decay', action='store_true', default=True)
parser.add_argument('--eval', action='store_true', default=False)
parser.add_argument('--adv_beta', type=float, default=1.0)
parser.add_argument('--threshold', type=float, default=-100)
parser.add_argument('--teacher', action='store_true', default=False)
parser.add_argument('--pis', action='store_true', default=False)
parser.add_argument('--percentile', type=float, default=0.95)
parser.add_argument('--on_policy_buffer', action='store_true', default=False)
parser.add_argument('--scheduler', action='store_true', default=True)
parser.add_argument('--mode', type=str, default='1')
parser.add_argument('--mix', action='store_true', default=True)
parser.add_argument('--alpha_main', type=float, default=0.5)
parser.add_argument('--rr', action='store_true', default=False)
parser.add_argument('--dim', type=int, default=2)
args = parser.parse_args()

set_seed(args.seed)
if 'SLURM_PROCID' in os.environ:
    args.seed += int(os.environ["SLURM_PROCID"])

eval_data_size = 2000
final_eval_data_size = 10000
plot_data_size = 2000
final_plot_data_size = 10000

if args.pis_architectures:
    args.zero_init = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
coeff_matrix = cal_subtb_coef_matrix(args.subtb_lambda, args.T).to(device)



def get_energy():
    if args.energy == '25gmm':
        energy = TwentyFiveGaussianMixture(device=device)
    elif args.energy == 'many_well':
        energy = ManyWell(device=device)
    return energy

def plot_step(energy, gfn_model, name, is_final=False):
    if args.energy == 'many_well':
        if is_final:
            batch_size = final_plot_data_size
        else:
            batch_size = plot_data_size
        samples = gfn_model.sample(batch_size, energy.log_reward)

        vizualizations = viz_many_well(energy, samples)
        fig_samples_x13, ax_samples_x13, fig_kde_x13, ax_kde_x13, fig_contour_x13, ax_contour_x13, fig_samples_x23, ax_samples_x23, fig_kde_x23, ax_kde_x23, fig_contour_x23, ax_contour_x23 = vizualizations

        return {"visualization/contourx13"+name: wandb.Image(fig_to_image(fig_contour_x13)),
                "visualization/contourx23"+name: wandb.Image(fig_to_image(fig_contour_x23)),
                "visualization/kdex13"+name: wandb.Image(fig_to_image(fig_kde_x13)),
                "visualization/kdex23"+name: wandb.Image(fig_to_image(fig_kde_x23)),
                "visualization/samplesx13"+name: wandb.Image(fig_to_image(fig_samples_x13)),
                "visualization/samplesx23"+name: wandb.Image(fig_to_image(fig_samples_x23))}

    elif energy.data_ndim != 2:
        return {}

    else:
        batch_size = plot_data_size
        samples = gfn_model.sample(batch_size, energy.log_reward)
        gt_samples = energy.sample(batch_size)

        fig_contour, ax_contour = get_figure(bounds=(-13., 13.))
        fig_kde, ax_kde = get_figure(bounds=(-13., 13.))
        fig_kde_overlay, ax_kde_overlay = get_figure(bounds=(-13., 13.))

        plot_contours(energy.log_reward, ax=ax_contour, bounds=(-13., 13.), n_contour_levels=150, device=device)
        plot_kde(gt_samples, ax=ax_kde_overlay, bounds=(-13., 13.))
        plot_kde(samples, ax=ax_kde, bounds=(-13., 13.))
        plot_samples(samples, ax=ax_contour, bounds=(-13., 13.))
        plot_samples(samples, ax=ax_kde_overlay, bounds=(-13., 13.))

        fig_contour.savefig(f'{name}contour.pdf', bbox_inches='tight')
        fig_kde_overlay.savefig(f'{name}kde_overlay.pdf', bbox_inches='tight')
        fig_kde.savefig(f'{name}kde.pdf', bbox_inches='tight')
        # return None
        return {"visualization/contour"+name: wandb.Image(fig_to_image(fig_contour)),
                "visualization/kde_overlay"+name: wandb.Image(fig_to_image(fig_kde_overlay)),
                "visualization/kde"+name: wandb.Image(fig_to_image(fig_kde))}


def eval_step(eval_data, energy, gfn_model, final_eval=False):
    gfn_model.eval()
    metrics = dict()
    if final_eval:
        init_state = torch.zeros(final_eval_data_size, energy.data_ndim).to(device)
        samples, metrics['final_eval/log_Z'], metrics['final_eval/log_Z_lb'], metrics[
            'final_eval/log_Z_learned'] = log_partition_function(
            init_state, gfn_model, energy.log_reward)
        if args.energy =='many_well':
            metrics['final_eval/delta_log_Z'] = 164.69567532 - metrics['final_eval/log_Z']
    else:
        init_state = torch.zeros(eval_data_size, energy.data_ndim).to(device)
        samples, metrics['eval/log_Z'], metrics['eval/log_Z_lb'], metrics[
            'eval/log_Z_learned'] = log_partition_function(
            init_state, gfn_model, energy.log_reward)
        if args.energy =='many_well':
            metrics['eval/delta_log_Z'] = 164.69567532 - metrics['eval/log_Z']
    if eval_data is None:
        log_elbo = None
        sample_based_metrics = None
    else:
        if final_eval:
            metrics['final_eval/mean_log_likelihood'] = 0. if args.mode_fwd == 'pis' else mean_log_likelihood(eval_data,
                                                                                                              gfn_model,
                                                                                                              energy.log_reward)
        else:
            metrics['eval/mean_log_likelihood'] = 0. if args.mode_fwd == 'pis' else mean_log_likelihood(eval_data,
                                                                                                        gfn_model,
                                                                                                        energy.log_reward)
        metrics.update(get_sample_metrics(samples, eval_data, final_eval))
        states, logpf, logpb, logf = gfn_model.get_trajectory_bwd(eval_data, None, energy.log_reward)
        gt_log_rewards = energy.log_reward(eval_data)
        gt_log_pfs = logpf.sum(-1)
        gt_log_pbs = logpb.sum(-1)
        if final_eval:
            metrics['final_eval/log_Z_ub'] = (gt_log_rewards + gt_log_pbs - gt_log_pfs).mean()
        else:
            metrics['eval/log_Z_ub'] = (gt_log_rewards + gt_log_pbs - gt_log_pfs).mean()

    gfn_model.train()
    return metrics



def adjust_backtrack_step(K, current_acceptance_rate, target_acceptance_rate=0.574):

    if current_acceptance_rate > target_acceptance_rate and K < 50:
        return K+1 
    elif current_acceptance_rate < target_acceptance_rate and K > 1:
        return K-1
    return K

def mcmc_step(energy, gfn_model, gfn_adv, init_log_pf, init_log_pb, init_state, init_reward, exploration_std, threshold, args, num_steps=20, K=20):
    accepted_samples = []
    accepted_logr = []
    acceptance_count = 0
    acceptance_rate = 0
    total_proposals = 0
    reward = energy.log_reward(init_state)
    for i in range(num_steps):
        if not args.teacher:
            s_prime = gfn_model.get_trajectory_back_and_forth(init_state, energy.log_reward, exploration_std, K)
        else:
            s_prime = gfn_adv.get_trajectory_back_and_forth(init_state, energy.log_reward, exploration_std, K)        
        _, states, log_pfs, log_pbs, log_r, adv_reward = bwd_tb(s_prime, gfn_model, energy.log_reward, exploration_std=None, 
        logr = None, return_exp=True, is_adv = False, threshold = threshold)

        log_pfs = log_pfs.sum(-1)
        log_pbs = log_pbs.sum(-1)

        forward_logps = init_log_pb + log_pfs
        backward_logps = init_log_pf + log_pbs
        

        
        if args.ls:
            accept_mask = energy.log_reward(s_prime) > energy.log_reward(init_state)
        elif args.els:
            accept_mask = adv_reward > init_reward
        
        acceptance_count += accept_mask.sum().item()
        total_proposals += s_prime.shape[0]
        acceptance_rate = acceptance_count / total_proposals
        K = adjust_backtrack_step(K, acceptance_rate, target_acceptance_rate=args.target_acceptance_rate)
        
        if i > int(num_steps/2):
            accepted_samples.append(s_prime[accept_mask])
            accepted_logr.append(log_r[accept_mask])

        init_state[accept_mask] = s_prime[accept_mask]
        init_reward[accept_mask] = adv_reward[accept_mask]
        init_log_pf[accept_mask] = log_pfs[accept_mask]
        init_log_pb[accept_mask] = log_pbs[accept_mask]
        improved_reward = energy.log_reward(init_state)
        # print(f"Step {i}, Acceptance rate: {acceptance_rate}, Reward: {improved_reward.mean().item()}")

    return init_state, torch.cat(accepted_samples, dim=0), torch.cat(accepted_logr, dim=0)





def train_step(energy, gfn_model, gfn_optimizer, gfn_adv, buffer, it, threshold, args):
    gfn_model.zero_grad()
    exploration_std = get_exploration_std(it, args.exploratory, args.exploration_factor, args.exploration_wd)


    # on-policy training
    if args.teacher:
        if it % 2 == 0:
            init_state = torch.zeros(args.batch_size, energy.data_ndim).to(device)
            loss, states, logpf, _, log_r, adv_reward  = fwd_tb(
                init_state,
                gfn_model,
                energy.log_reward,
                exploration_std=exploration_std,
                return_exp=True,
                threshold=threshold,
            )
            if it == 0:
                buffer.add(states[:, -1], adv_reward)
            if args.els and it % (50) == 0:
                samples, rewards = buffer.sample()
                local_search_samples, log_r = langevin_dynamics(samples, energy.log_reward, device, args)
                buffer.add(local_search_samples, log_r)

        # off-policy training
        else:
            if it % (2*args.adv_freq) == 1:
                loss, states, adv_reward = bwd_train_step(
                    energy, gfn_model, gfn_adv, buffer, exploration_std, threshold, args, it=it, is_buffer=False
                )
                #buffer.add(states[:, -1], energy.log_reward(states[:, -1]))
            else:
                loss, states, adv_reward = bwd_train_step(
                    energy, gfn_model, gfn_adv, buffer, exploration_std, threshold, args, it=it, is_buffer=True
                )

    elif args.per:
        if it % 2 == 0:
            init_state = torch.zeros(args.batch_size, energy.data_ndim).to(device)
            loss, states, logpf, _, log_r, adv_reward  = fwd_tb(
                init_state,
                gfn_model,
                energy.log_reward, 
                exploration_std=exploration_std,
                return_exp=True,
                threshold=threshold,
            )
            if it == 0:
                buffer.add(states[:, -1], adv_reward)
            if it % (50) == 0:
                samples, rewards = buffer.sample()
                local_search_samples, log_r = langevin_dynamics(samples, energy.log_reward, device, args)
                buffer.add(local_search_samples, log_r)            
            else:
                filtering = energy.log_reward(states[:, -1]) > threshold
                states = states[filtering]
                log_r = log_r[filtering]
                adv_reward = adv_reward[filtering]
                buffer.add(states[:, -1], adv_reward)
        
        # off-policy training of PER
        else:
            samples, rewards = buffer.sample()
            loss, states, _, _, _, adv_reward = bwd_tb(
                samples, gfn_model, energy.log_reward, exploration_std, return_exp=True, is_adv=False, threshold=threshold
            )

                #buffer.add(states[:, -1], energy.log_reward(states[:, -1]))
   
    else:
        init_state = torch.zeros(args.batch_size, energy.data_ndim).to(device)
        loss, states, logpf, _, log_r, adv_reward  = fwd_tb(
            init_state,
            gfn_model,
            energy.log_reward,
            exploration_std=exploration_std,
            return_exp=True,
            threshold=threshold,
        )


    loss.backward()
    gfn_optimizer.step()
    return loss.item(), states[:, -1], adv_reward, threshold



def train_step_adv(energy, gfn_adv, gfn_adv_optimizer, samples, adv_reward, it, threshold, args):
    gfn_adv.zero_grad()

    states, log_pfs, logpb, log_fs = gfn_adv.get_trajectory_bwd(samples.detach(), None, energy.log_reward)
    logr = energy.log_reward(states[:, -1])
    filtering = logr > threshold
    log_pfs = log_pfs.sum(-1)
    log_pfs = log_pfs[filtering]
    logpb = logpb.sum(-1)
    logpb = logpb[filtering]
    logz = log_fs[filtering, 0]
    adv_reward = adv_reward[filtering]
    log_adv = torch.log(adv_reward)
    logr = logr[filtering]

    if args.mix:
        log_reward = log_adv + args.alpha_main * logr
    else:
        log_reward = log_adv


    try:
        loss = 0.5 * ((log_pfs + logz - logpb - log_reward) ** 2)
        loss = loss.mean()
        loss.backward()
        max_norm = 0.1  
        torch.nn.utils.clip_grad_norm_(gfn_adv_optimizer.param_groups[0]['params'], max_norm)
        gfn_adv_optimizer.step()
        return loss.item()
    except:
        return 0

def bwd_train_step(energy, gfn_model, gfn_adv, buffer, exploration_std, threshold, args, it, is_buffer = True):

    if is_buffer:
        samples, adv_rewards = buffer.sample()

    else:
        samples = gfn_adv.sample(args.batch_size, energy.log_reward)
        filtering = energy.log_reward(samples) > threshold
        samples = samples[filtering]
        loss, states, log_pfs, log_pbs, log_r, adv_reward = bwd_tb(
            samples, gfn_model, energy.log_reward, exploration_std, return_exp=True, is_adv=False, threshold=threshold
        )
        buffer.add(states[:, -1], adv_reward)
        samples, adv_rewards = buffer.sample()

    loss, states, log_pfs, log_pbs, log_r, adv_reward = bwd_tb(
        samples, gfn_model, energy.log_reward, exploration_std, return_exp=True, is_adv=False, threshold=threshold
    )

    return loss, states, adv_reward

def train():
    name = get_name(args)
    if not os.path.exists(name):
        os.makedirs(name)

    energy = get_energy()
    eval_data = energy.sample(eval_data_size).to(device)

    config = args.__dict__
    config["Experiment"] = "{args.energy}"
    if WANDB:
        if args.teacher:
            wandb.init(project="diffusion sampling", config=config, name='teacher/' + 'dim: ' + str(args.dim) + '/' + 'h_dim: ' + str(args.hidden_dim) + '/' + 'b_size' + str(args.buffer_size))
        elif args.per:
            wandb.init(project="diffusion sampling", config=config, name='per/' + 'dim: ' + str(args.dim) + '/' + 'h_dim: ' + str(args.hidden_dim) + '/' + 'b_size' + str(args.buffer_size))
        else:
            wandb.init(project="diffusion sampling", config=config, name='none/' + 'dim: ' + str(args.dim) + '/' + 'h_dim: ' + str(args.hidden_dim) + '/' + 'b_size' + str(args.buffer_size))


    gfn_model = GFN(energy.data_ndim, args.s_emb_dim, args.hidden_dim, args.harmonics_dim, args.t_emb_dim,
                    trajectory_length=args.T, clipping=args.clipping, lgv_clip=args.lgv_clip, gfn_clip=args.gfn_clip,
                    langevin=args.langevin, learned_variance=args.learned_variance,
                    partial_energy=args.partial_energy, log_var_range=args.log_var_range,
                    pb_scale_range=args.pb_scale_range,
                    t_scale=args.t_scale, langevin_scaling_per_dimension=args.langevin_scaling_per_dimension,
                    conditional_flow_model=args.conditional_flow_model, learn_pb=args.learn_pb,
                    pis_architectures=args.pis_architectures, lgv_layers=args.lgv_layers,
                    joint_layers=args.joint_layers, zero_init=args.zero_init, device=device).to(device)


    gfn_adv = GFN(energy.data_ndim, args.s_emb_dim, args.hidden_dim, args.harmonics_dim, args.t_emb_dim,
                    trajectory_length=args.T, clipping=args.clipping, lgv_clip=args.lgv_clip, gfn_clip=args.gfn_clip,
                    langevin=False, learned_variance=args.learned_variance,
                    partial_energy=args.partial_energy, log_var_range=4.0,
                    pb_scale_range=args.pb_scale_range,
                    t_scale=args.t_scale, langevin_scaling_per_dimension=args.langevin_scaling_per_dimension,
                    conditional_flow_model=args.conditional_flow_model, learn_pb=args.learn_pb,
                    pis_architectures=args.pis_architectures, lgv_layers=args.lgv_layers,
                    joint_layers=args.joint_layers, zero_init=args.zero_init, device=device).to(device)


    gfn_optimizer = get_gfn_optimizer(gfn_model, args.lr_policy, args.lr_flow, args.lr_back, args.learn_pb,
                                      args.conditional_flow_model, args.use_weight_decay, args.weight_decay)

    gfn_adv_optimizer = get_gfn_optimizer(gfn_adv, 5*args.lr_policy, args.lr_flow, 5*args.lr_back, args.learn_pb,
                                      args.conditional_flow_model, args.use_weight_decay, args.weight_decay)


    gamma = 0.1

    scheduler = MultiStepLR(gfn_optimizer, milestones=[args.epochs - 6000, args.epochs - 2000], gamma=gamma)
    scheduler_adv = MultiStepLR(gfn_adv_optimizer, milestones=[args.epochs - 6000, args.epochs - 2000], gamma=gamma)
    metrics = dict()

    buffer = ReplayBuffer(args.buffer_size, device, energy.log_reward,args.batch_size, args, data_ndim=energy.data_ndim, beta=args.beta,
                          rank_weight=args.rank_weight, prioritized=args.prioritized)

    gfn_model.train()
    gfn_adv.train()
    samples = gfn_model.sample(args.batch_size, energy.log_reward)
    log_r = energy.log_reward(samples)
    threshold = torch.quantile(log_r, 0.9).item()


    for i in trange(args.epochs + 1):
        
  
        metrics['train/loss'], samples, adv_reward, threshold = train_step(energy, gfn_model, gfn_optimizer, gfn_adv, buffer, i, threshold, args)
        
        metrics['train/lower_bound'] = threshold

        if not args.teacher:
            metrics['train/adv_loss'] = 0
        else:
            metrics['train/adv_loss'] = train_step_adv(energy, gfn_adv, gfn_adv_optimizer, samples, adv_reward, i, threshold, args)

        if args.scheduler:
            scheduler.step()
            if args.teacher:
                scheduler_adv.step()

        if WANDB and i % 100 == 0:
            metrics.update(eval_step(eval_data, energy, gfn_model, final_eval=False))
            if 'tb-avg' in args.mode_fwd or 'tb-avg' in args.mode_bwd:
                del metrics['eval/log_Z_learned']
            images = plot_step(energy, gfn_model,  name + "student gfn")
            metrics.update(images)
            images_adv = plot_step(energy, gfn_adv, name + "teacher gfn")
            metrics.update(images_adv)

            plt.close('all')
            wandb.log(metrics, step=i)


    eval_results = final_eval(energy, gfn_model)
    metrics.update(eval_results)
    if 'tb-avg' in args.mode_fwd or 'tb-avg' in args.mode_bwd:
        del metrics['eval/log_Z_learned']
    images = plot_step(energy, gfn_model,  name + "student gfn", is_final=True)
    metrics.update(images)
    images_adv = plot_step(energy, gfn_adv, name + "teacher gfn", is_final=True)
    metrics.update(images_adv)
    plt.close('all')
    wandb.log(metrics, step=1)
    wandb.finish()
    # torch.save(gfn_model.state_dict(), f'{name}model_final.pt')


def final_eval(energy, gfn_model):
    final_eval_data = energy.sample(final_eval_data_size).to(device)
    results = eval_step(final_eval_data, energy, gfn_model, final_eval=True)
    return results


if __name__ == '__main__':
        train()
 