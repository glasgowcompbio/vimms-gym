# modified from:
# - https://github.com/optuna/optuna-examples/blob/main/rl/sb3_simple.py
# - https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/utils/hyperparams_opt.py

import os
from timeit import default_timer as timer
from datetime import timedelta

import numpy as np
import torch.nn as nn
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import sync_envs_normalization
from vimms.Evaluation import EvaluationData

from vimms_gym.common import linear_schedule, EVAL_METRIC_REWARD, EVAL_METRIC_F1, \
    MAX_EVAL_TIME_PER_EPISODE, evaluate, METHOD_PPO


def sample_dqn_params(trial, tune_model, tune_reward):
    params = {}

    if tune_model:
        gamma = trial.suggest_categorical('gamma', [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1, log=True)
        lr_schedule = trial.suggest_categorical('lr_schedule', ['linear', 'constant'])
        if lr_schedule == 'linear':
            learning_rate = linear_schedule(learning_rate)

        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 100, 128, 256, 512])
        buffer_size = trial.suggest_categorical('buffer_size',
                                                [int(1e4), int(5e4), int(1e5), int(1e6)])
        exploration_final_eps = trial.suggest_float('exploration_final_eps', 0, 0.2)
        exploration_fraction = trial.suggest_float('exploration_fraction', 0, 0.5)
        target_update_interval = trial.suggest_categorical('target_update_interval',
                                                            [1, 1000, 5000, 10000, 15000, 20000])
        learning_starts = trial.suggest_categorical('learning_starts',
                                                    [0, 1000, 5000, 10000, 20000])

        train_freq = trial.suggest_categorical('train_freq', [1, 4, 8, 16, 128, 256, 1000])
        subsample_steps = trial.suggest_categorical('subsample_steps', [1, 2, 4, 8])
        gradient_steps = max(train_freq // subsample_steps, 1)

        net_arch = trial.suggest_categorical('net_arch', ['small', 'medium', 'large'])
        net_arch = {
            'small': [64, 64],
            'medium': [256, 256],
            'large': [512, 512]
        }[net_arch]

        params.update({
            'gamma': gamma,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'buffer_size': buffer_size,
            'train_freq': train_freq,
            'gradient_steps': gradient_steps,
            'exploration_fraction': exploration_fraction,
            'exploration_final_eps': exploration_final_eps,
            'target_update_interval': target_update_interval,
            'learning_starts': learning_starts,
            'policy_kwargs': dict(net_arch=net_arch),
        })

    if tune_reward:
        alpha = trial.suggest_uniform('alpha', 0, 1)
        beta = trial.suggest_uniform('beta', 0, 1)
        params.update({
            'alpha': alpha,
            'beta': beta,
        })

    return params


def sample_ppo_params(trial, tune_model, tune_reward):
    params = {}

    if tune_model:
        batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64, 128, 256, 512])
        n_steps = trial.suggest_categorical('n_steps', [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
        gamma = trial.suggest_categorical('gamma', [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1, log=True)
        lr_schedule = trial.suggest_categorical('lr_schedule', ['linear', 'constant'])
        ent_coef = trial.suggest_float('ent_coef', 0.00000001, 0.1, log=True)
        clip_range = trial.suggest_categorical('clip_range', [0.1, 0.2, 0.3, 0.4])
        n_epochs = trial.suggest_categorical('n_epochs', [1, 5, 10, 20])
        gae_lambda = trial.suggest_categorical('gae_lambda',
                                                [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
        max_grad_norm = trial.suggest_categorical('max_grad_norm',
                                                    [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
        vf_coef = trial.suggest_float('vf_coef', 0, 1)

        # Uncomment for gSDE (continuous actions)
        # log_std_init = trial.suggest_float('log_std_init', -4, 1)

        # Uncomment for gSDE (continuous action)
        # sde_sample_freq = trial.suggest_categorical('sde_sample_freq', [-1, 8, 16, 32, 64, 128, 256])

        # Orthogonal initialization
        ortho_init = False
        # ortho_init = trial.suggest_categorical('ortho_init', [False, True])

        # activation_fn = trial.suggest_categorical('activation_fn',
        #                                           ['tanh', 'relu', 'elu', 'leaky_relu'])
        activation_fn = trial.suggest_categorical('activation_fn', ['tanh', 'relu'])

        # TODO: account when using multiple envs
        if batch_size > n_steps:
            batch_size = n_steps

        if lr_schedule == 'linear':
            learning_rate = linear_schedule(learning_rate)

        # Independent networks usually work best
        # when not working with images
        net_arch = trial.suggest_categorical('net_arch', ['small', 'medium', 'large'])
        net_arch = {
            'small': [dict(pi=[64, 64], vf=[64, 64])],
            'medium': [dict(pi=[256, 256], vf=[256, 256])],
            'large': [dict(pi=[512, 512], vf=[512, 512])],
        }[net_arch]

        activation_fn = \
        {'tanh': nn.Tanh, 'relu': nn.ReLU, 'elu': nn.ELU, 'leaky_relu': nn.LeakyReLU}[
            activation_fn]

        params.update({
            'n_steps': n_steps,
            'batch_size': batch_size,
            'gamma': gamma,
            'learning_rate': learning_rate,
            'ent_coef': ent_coef,
            'clip_range': clip_range,
            'n_epochs': n_epochs,
            'gae_lambda': gae_lambda,
            'max_grad_norm': max_grad_norm,
            'vf_coef': vf_coef,
            # 'sde_sample_freq': sde_sample_freq,
            'policy_kwargs': dict(
                # log_std_init=log_std_init,
                net_arch=net_arch,
                activation_fn=activation_fn,
                ortho_init=ortho_init,
            ),
        })

    if tune_reward:
        alpha = trial.suggest_uniform('alpha', 0, 1)
        beta = trial.suggest_uniform('beta', 0, 1)
        params.update({
            'alpha': alpha,
            'beta': beta,
        })

    return params


class TrialEvalCallback(EvalCallback):
    def __init__(self, eval_env, model_name, trial, eval_metric, n_eval_episodes, eval_freq,
                 max_eval_time_per_episode, deterministic=True, verbose=0,
                 best_model_save_path=None, log_path=None):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
            best_model_save_path=best_model_save_path,
            log_path=log_path
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False
        self.eval_metric = eval_metric
        self.max_eval_time_per_episode = max_eval_time_per_episode
        self.model_name = model_name

    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            # warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")
            pass

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # Init callback called on new best model
        if self.callback_on_new_best is not None:
            self.callback_on_new_best.init_callback(self.model)

    def _on_step(self):
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if self.verbose > 0:
                print('Evaluation %d called' % self.eval_idx)
            start = timer()
            self.custom_on_step()
            end = timer()
            if self.verbose > 0:
                print('Evaluation %d finished: last_mean_reward %f, timedelta=%s' % (
                    self.eval_idx, self.last_mean_reward, str(timedelta(seconds=end - start))))

            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True

    def custom_on_step(self):

        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:

            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    )

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = self.custom_evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=self.deterministic
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose > 0:
                print(
                    f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")

            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training

    def custom_evaluate_policy(self, model, env, n_eval_episodes=10, deterministic=True):

        episode_count = 0
        episode_count_target = n_eval_episodes
        current_reward = 0
        current_length = 0
        observations = env.reset()
        states = None
        episode_starts = np.ones((env.num_envs,), dtype=bool)


        episode_rewards = []
        episode_lengths = []
        start = timer()
        while episode_count < episode_count_target:
            inner_env = env.envs[0].env

            if self.model_name == METHOD_PPO: # maskable PPO
                action_masks = inner_env.action_masks()
                actions, states = model.predict(observations, state=states,
                                                episode_start=episode_starts,
                                                deterministic=deterministic,
                                                action_masks=action_masks)
            else: # DQN doesn't use action masking yet
                actions, states = model.predict(observations, state=states,
                                                episode_start=episode_starts,
                                                deterministic=deterministic)

            observations, rewards, dones, infos = env.step(actions)
            episode_starts = dones
            current_reward += rewards[0]
            current_length += 1

            if dones[0]:  # when done, episode would be reset automatically
                if self.eval_metric == EVAL_METRIC_REWARD:
                    val = current_reward
                else:
                    eval_res = evaluate(eval_data, format_output=False)
                    assert self.eval_metric in eval_res, 'Unknown evaluation metric'
                    val = eval_res[self.eval_metric]

                end = timer()
                delta = end - start
                if self.verbose > 0:
                    print('Evaluation episode %d finished: metric %f, timedelta=%s' % (
                        episode_count, val, str(timedelta(seconds=delta))))                    
                if delta > self.max_eval_time_per_episode:
                    raise ValueError('Eval time per episode (%.2f seconds) '\
                        'exceeds the budget of %.2f seconds' % (delta, self.max_eval_time_per_episode))
                start = timer()
                episode_rewards.append(val)
                episode_lengths.append(current_length)
                episode_count += 1
                current_reward = 0
                current_length = 0

            # store previous results for evaluation before 'done'
            # this needs to be here, because VecEnv is automatically reset when done
            eval_data = EvaluationData(inner_env.vimms_env)

        return episode_rewards, episode_lengths
