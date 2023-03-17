import os
from os.path import exists

import numpy as np
from loguru import logger
import torch.nn as nn

from vimms.ChemicalSamplers import MZMLFormulaSampler, MZMLRTandIntensitySampler, \
    GaussianChromatogramSampler, MZMLChromatogramSampler
from vimms.Common import POSITIVE, load_obj, save_obj
from vimms.Roi import RoiBuilderParams
from vimms_gym.common import METHOD_DQN, METHOD_PPO, ALPHA, BETA
from vimms_gym.common import linear_schedule

ENV_QCB_SMALL_GAUSSIAN = 'QCB_chems_small'
ENV_QCB_MEDIUM_GAUSSIAN = 'QCB_chems_medium'
ENV_QCB_LARGE_GAUSSIAN = 'QCB_chems_large'

ENV_QCB_SMALL_EXTRACTED = 'QCB_resimulated_small'
ENV_QCB_MEDIUM_EXTRACTED = 'QCB_resimulated_medium'
ENV_QCB_LARGE_EXTRACTED = 'QCB_resimulated_large'


def get_qcb_filename():
    # for running from script
    from_notebook = False
    base_dir = '.'
    mzml_filename = os.path.abspath(os.path.join(base_dir, 'notebooks', 'fullscan_QCB.mzML'))

    # for running from notebooks
    if not exists(mzml_filename):
        from_notebook = True
        base_dir = '../..'
        mzml_filename = os.path.abspath(os.path.join(base_dir, 'notebooks', 'fullscan_QCB.mzML'))

    assert exists(mzml_filename), '%s is missing' % mzml_filename
    return base_dir, mzml_filename, from_notebook


def preset_qcb_small(model_name, alpha=ALPHA, beta=BETA, extract_chromatograms=False):
    max_peaks = 100
    base_dir, mzml_filename, from_notebook = get_qcb_filename()
    samplers_pickle_prefix = 'samplers_QCB_small'
    n_chemicals = (20, 50)
    mz_range = (100, 110)
    rt_range = (400, 500)
    intensity_range = (1E4, 1E20)

    params = generate_params(mzml_filename, samplers_pickle_prefix, n_chemicals,
                             mz_range, rt_range, intensity_range, extract_chromatograms)
    params['env']['alpha'] = alpha
    params['env']['beta'] = beta

    if model_name == METHOD_DQN:

        # FIXME: this is WRONG. This is for preset_qcb_medium
        # And anyway we should delete this preset as it's too easy
        # most stable reward for 1E6 timesteps
        # alpha   = 0.25
        # beta    = 0.50
        # horizon = 4
        gamma = 0.9
        learning_rate = 0.000232586
        batch_size = 128
        buffer_size = 100000
        train_freq = 128
        subsample_steps = 1
        gradient_steps = max(train_freq // subsample_steps, 1)
        exploration_fraction = 0.450071463
        exploration_final_eps = 0.060611843
        target_update_interval = 5000
        learning_starts = 1000
        hidden_nodes = 64

        policy_kwargs = dict(net_arch=[hidden_nodes, hidden_nodes])
        params['model'] = {
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
            'policy_kwargs': policy_kwargs
        }
    elif model_name == METHOD_PPO:
        # FIXME: not sure where this came from, need to optimise
        hidden_nodes = 512
        net_arch = [dict(pi=[hidden_nodes, hidden_nodes], vf=[hidden_nodes, hidden_nodes])]
        params['model'] = {
            'n_steps': 2048,
            'batch_size': 512,
            'gamma': 0.90,
            'learning_rate': 0.0001,
            'ent_coef': 0.001,
            'gae_lambda': 0.90,
            'policy_kwargs': dict(
                net_arch=net_arch,
            ),
        }
    return params, max_peaks


def preset_qcb_medium(model_name, alpha=ALPHA, beta=BETA, extract_chromatograms=False):
    max_peaks = 200
    base_dir, mzml_filename, from_notebook = get_qcb_filename()
    samplers_pickle_prefix = 'samplers_QCB_medium'
    n_chemicals = (200, 500)
    mz_range = (100, 600)
    rt_range = (400, 800)
    intensity_range = (1E4, 1E20)

    params = generate_params(mzml_filename, samplers_pickle_prefix, n_chemicals,
                             mz_range, rt_range, intensity_range, extract_chromatograms)
    params['env']['alpha'] = alpha
    params['env']['beta'] = beta

    if model_name == METHOD_DQN:

        # best reward with 1E6 timesteps
        # alpha   = 0.25
        # beta    = 0.50
        # horizon = 4
        gamma = 0.95
        learning_rate = 0.00014450137513290646
        batch_size = 256
        buffer_size = 10000
        train_freq = 1
        subsample_steps = 2
        gradient_steps = max(train_freq // subsample_steps, 1)
        exploration_fraction = 0.27797033409246663
        exploration_final_eps = 0.005589071654866951
        target_update_interval = 20000
        learning_starts = 2000
        hidden_nodes = 256
        policy_kwargs = dict(net_arch=[hidden_nodes, hidden_nodes])
        params['model'] = {
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
            'policy_kwargs': policy_kwargs
        }

    elif model_name == METHOD_PPO:

        # top-5 best reward with 1E5 timesteps, but the fastest (~5 mins)
        # alpha   = 0.25
        # beta    = 0.50
        # horizon = 4

        n_steps = 512
        batch_size = 512
        gamma = 0.90
        learning_rate = 0.557248151
        lr_schedule = 'linear'
        ent_coef = 5.40E-08
        clip_range = 0.3
        n_epochs = 1
        gae_lambda = 0.98
        max_grad_norm = 0.5
        vf_coef = 0.524370133
        ortho_init = False
        activation_fn = 'relu'
        net_arch = 'large'

        if lr_schedule == 'linear':
            learning_rate = linear_schedule(learning_rate)

        activation_fn = \
        {'tanh': nn.Tanh, 'relu': nn.ReLU, 'elu': nn.ELU, 'leaky_relu': nn.LeakyReLU}[
            activation_fn]

        net_arch = {
            'small': [dict(pi=[64, 64], vf=[64, 64])],
            'medium': [dict(pi=[256, 256], vf=[256, 256])],
            'large': [dict(pi=[512, 512], vf=[512, 512])],
        }[net_arch]

        params['model'] = {
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
        }

    return params, max_peaks


def preset_qcb_large(model_name, alpha=ALPHA, beta=BETA, extract_chromatograms=False):
    max_peaks = 200
    base_dir, mzml_filename, from_notebook = get_qcb_filename()
    samplers_pickle_prefix = 'samplers_QCB_large'
    n_chemicals = (2000, 5000)
    mz_range = (70, 1000)
    rt_range = (0, 1440)
    intensity_range = (1E4, 1E20)

    params = generate_params(mzml_filename, samplers_pickle_prefix, n_chemicals,
                             mz_range, rt_range, intensity_range, extract_chromatograms,
                             at_least_one_point_above=1E5)
    params['env']['alpha'] = alpha
    params['env']['beta'] = beta

    if model_name == METHOD_DQN:

        # same as preset_qcb_medium()
        # alpha   = 0.25
        # beta    = 0.50
        # horizon = 4
        gamma = 0.95
        learning_rate = 0.00014450137513290646
        batch_size = 256
        buffer_size = 10000
        train_freq = 1
        subsample_steps = 2
        gradient_steps = max(train_freq // subsample_steps, 1)
        exploration_fraction = 0.27797033409246663
        exploration_final_eps = 0.005589071654866951
        target_update_interval = 20000
        learning_starts = 2000
        hidden_nodes = 256
        policy_kwargs = dict(net_arch=[hidden_nodes, hidden_nodes])
        params['model'] = {
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
            'policy_kwargs': policy_kwargs
        }
    elif model_name == METHOD_PPO:

        # same as preset_qcb_medium()
        # alpha   = 0.25
        # beta    = 0.50
        # horizon = 4

        n_steps = 512
        batch_size = 512
        gamma = 0.90
        learning_rate = 0.557248151
        lr_schedule = 'linear'
        ent_coef = 5.40E-08
        clip_range = 0.3
        n_epochs = 1
        gae_lambda = 0.98
        max_grad_norm = 0.5
        vf_coef = 0.524370133
        ortho_init = False
        activation_fn = 'relu'
        net_arch = 'large'

        if lr_schedule == 'linear':
            learning_rate = linear_schedule(learning_rate)

        activation_fn = \
        {'tanh': nn.Tanh, 'relu': nn.ReLU, 'elu': nn.ELU, 'leaky_relu': nn.LeakyReLU}[
            activation_fn]

        net_arch = {
            'small': [dict(pi=[64, 64], vf=[64, 64])],
            'medium': [dict(pi=[256, 256], vf=[256, 256])],
            'large': [dict(pi=[512, 512], vf=[512, 512])],
        }[net_arch]

        params['model'] = {
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
        }

    return params, max_peaks


def generate_params(mzml_filename, samplers_pickle_prefix, n_chemicals, mz_range, rt_range,
                    intensity_range, extract_chromatograms, isolation_window=0.7, mz_tol=10,
                    rt_tol=5, use_dew=False, min_ms1_intensity=5000,
                    ionisation_mode=POSITIVE, enable_spike_noise=True,
                    noise_density=0.1, noise_max_val=1E3,
                    min_roi_length=3, at_least_one_point_above=5E5):
    samplers_pickle_suffix = 'extracted' if extract_chromatograms else 'gaussian'

    base_dir, mzml_filename, from_notebook = get_qcb_filename()

    pickle_base_dir = base_dir # from notebook
    if not from_notebook: # from script
        pickle_base_dir = os.path.join(base_dir, '..')

    samplers_pickle = os.path.abspath(os.path.join(pickle_base_dir, 'pickles', '%s_%s.p' % (
        samplers_pickle_prefix, samplers_pickle_suffix)))

    min_mz = mz_range[0]
    max_mz = mz_range[1]
    min_rt = rt_range[0]
    max_rt = rt_range[1]
    min_log_intensity = np.log(intensity_range[0])
    max_log_intensity = np.log(intensity_range[1])
    mz_sampler, ri_sampler, cr_sampler = get_samplers(mzml_filename, samplers_pickle, min_mz,
                                                      max_mz, min_rt, max_rt, min_log_intensity,
                                                      max_log_intensity, extract_chromatograms,
                                                      min_roi_length, at_least_one_point_above)
    params = {
        'chemical_creator': {
            'mz_range': mz_range,
            'rt_range': rt_range,
            'intensity_range': intensity_range,
            'n_chemicals': n_chemicals,
            'mz_sampler': mz_sampler,
            'ri_sampler': ri_sampler,
            'cr_sampler': cr_sampler,
        },
        'noise': {
            'enable_spike_noise': enable_spike_noise,
            'noise_density': noise_density,
            'noise_max_val': noise_max_val,
            'mz_range': mz_range
        },
        'env': {
            'ionisation_mode': ionisation_mode,
            'rt_range': rt_range,
            'isolation_window': isolation_window,
            'use_dew': use_dew,
            'mz_tol': mz_tol,
            'rt_tol': rt_tol,
            'min_ms1_intensity': min_ms1_intensity
        }
    }
    return params


def get_samplers(mzml_filename, samplers_pickle, min_mz, max_mz, min_rt, max_rt, min_log_intensity,
                 max_log_intensity, extract_chromatograms, min_roi_length,
                 at_least_one_point_above):
    if exists(samplers_pickle):
        logger.info('Loaded %s' % samplers_pickle)
        samplers = load_obj(samplers_pickle)
        mz_sampler = samplers['mz']
        ri_sampler = samplers['rt_intensity']
        cr_sampler = samplers['chromatogram']
    else:
        logger.info('Creating samplers from %s' % mzml_filename)
        mz_sampler = MZMLFormulaSampler(mzml_filename, min_mz=min_mz, max_mz=max_mz)
        ri_sampler = MZMLRTandIntensitySampler(mzml_filename, min_rt=min_rt, max_rt=max_rt,
                                               min_log_intensity=min_log_intensity,
                                               max_log_intensity=max_log_intensity)
        if extract_chromatograms:
            roi_params = RoiBuilderParams(min_roi_length=min_roi_length,
                                          at_least_one_point_above=at_least_one_point_above)
            cr_sampler = MZMLChromatogramSampler(mzml_filename, roi_params=roi_params)
        else:
            cr_sampler = GaussianChromatogramSampler()
        samplers = {
            'mz': mz_sampler,
            'rt_intensity': ri_sampler,
            'chromatogram': cr_sampler
        }
        save_obj(samplers, samplers_pickle)
    return mz_sampler, ri_sampler, cr_sampler
