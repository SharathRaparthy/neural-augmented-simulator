try:
    from comet_ml import Experiment
    comet_loaded = False
except ImportError:
    comet_loaded = False

import os
import time
from collections import deque

import numpy as np
import torch
import gym
from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy, RandomPolicy, NaviBase
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate
from nas.models.networks import LstmNetRealv1


def main():
    args = get_args()
    HIDDEN_NODES = 128
    LSTM_LAYERS = 3

    if comet_loaded and len(args.comet) > 0:
        comet_credentials = args.comet.split("/")
        experiment = Experiment(
            api_key=comet_credentials[2],
            project_name=comet_credentials[1],
            workspace=comet_credentials[0])
        for key, value in vars(args).items():
            experiment.log_parameter(key, value)
    else:
        experiment = None

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False,
                         args.custom_gym, args.navi)

    base = None
    if args.navi:
        base = NaviBase
    obs_shape = envs.observation_space.shape

    actor_critic = Policy(
        obs_shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy},
        navi=args.navi,
        base=base,
    )
    actor_critic.to(device)
    lstm_net = LstmNetRealv1(
        n_input_state_sim=12,
        n_input_state_real=12,
        n_input_actions=6,
        nodes=HIDDEN_NODES,
        layers=LSTM_LAYERS)
    lstm_net.to(device)
    file_path = '/home/sharath/neural-augmented-simulator/nas/data/trained_models/model-exp1-h128-l3-v10-e5.pth'
    lstm_net.load_state_dict(torch.load(file_path))

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'random':
        agent = algo.RANDOM_AGENT(actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

        actor_critic = RandomPolicy(obs_shape,
                                    envs.action_space,
                                    base_kwargs={'recurrent': args.recurrent_policy},
                                    navi=args.navi,
                                    base=base,
                                    )
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    if args.gail:
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
            device)
        file_name = os.path.join(
            args.gail_experts_dir,
            "trajs_{}.pt".format(args.env_name.split('-')[0].lower()))

        gail_train_loader = torch.utils.data.DataLoader(
            gail.ExpertDataset(
                file_name, num_trajectories=4, subsample_frequency=20),
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=True)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    episode_length = deque(maxlen=10)
    episode_success_rate = deque(maxlen=100)
    episode_total = 0

    start = time.time()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        print("args.num_steps: " + str(args.num_steps))
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])
            #Incorporating LSTM in the PPO loop ---------------------------
            # TODO: Normalize the observations and actions coz some values are greater than 1. Check with Florian.
            last_obs = torch.zeros(1, 12).float()
            last_obs[:, [1, 2, 4, 5, 7, 8, 10, 11]] = obs[:, :8].cpu()
            # Observe reward and next obs
            new_obs, reward, done, infos = envs.step(action)
            action_input = np.insert(np.insert(action.cpu().numpy(), 0, 0), 3, 0)
            obs_modified = torch.zeros(1,12).float()
            obs_modified[:, [1, 2, 4, 5, 7, 8, 10, 11]] = new_obs[:, :8].cpu()
            input_tensor = torch.FloatTensor(np.hstack((last_obs.numpy(), np.expand_dims(action_input, axis=0),
                                                        obs_modified.numpy()))).unsqueeze(0).to(device) # Unsqueeze because LSTM requires 3 dim input

            '''Checking the outputs'''
            # print(f'The action produced by PPO is {action}')
            # print(f'The action input to the LSTM is {action_input}')
            # print(f"The observation before is {obs}")
            # print(f'The last observation is {last_obs}')
            # print(f"The input to the LSTM is {input_tensor}")
            # print(f'The modified observation is {obs_modified}')
            with torch.no_grad():
                diff = lstm_net.forward(input_tensor)
            pred_obs = torch.FloatTensor(diff.cpu().numpy() + input_tensor.cpu().numpy()[:, :, 18:]).squeeze(0) # squeeze because PPO takes 2 dim input
            new_obs[:, :8] = pred_obs[:, [1, 2, 4, 5, 7, 8, 10, 11]]
            obs = new_obs.to(device)

            for idx, info in enumerate(infos):
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    episode_length.append(info['episode']['l'])
                    # if "Pacman" not in args.env_name:
                    #     episode_success_rate.append(info['was_successful_trajectory'])
                    episode_total += 1

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        if args.gail:
            if j >= 10:
                envs.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts,
                             utils.get_vec_normalize(envs)._obfilt)

            for step in range(args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], args.gamma,
                    rollouts.masks[step])

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0 or
                j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            if experiment is not None:
                experiment.log_metric("Reward Mean", np.mean(episode_rewards), step=total_num_steps)
                experiment.log_metric("Reward Min", np.min(episode_rewards), step=total_num_steps)
                experiment.log_metric("Reward Max", np.max(episode_rewards), step=total_num_steps)
                experiment.log_metric("Episode Length Mean ", np.mean(episode_length), step=total_num_steps)
                experiment.log_metric("Episode Length Min", np.min(episode_length), step=total_num_steps)
                experiment.log_metric("Episode Length Max", np.max(episode_length), step=total_num_steps)
                experiment.log_metric("# Trajectories (Total)", j, step=total_num_steps)
                # if "Pacman" not in args.env_name:
                #     experiment.log_metric("Episodic Success Rate", np.mean(episode_success_rate), step=total_num_steps)
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))

        if (args.eval_interval is not None and len(episode_rewards) > 1 and
                j % args.eval_interval == 0):
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            evaluate(actor_critic, ob_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)


if __name__ == "__main__":
    main()
