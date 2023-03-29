import copy
import sys

sys.path.append('../..')

import pandas as pd

from vimms.Common import set_log_level_warning, load_obj
from vimms.Noise import UniformSpikeNoise
from vimms.MassSpec import IndependentMassSpectrometer
from vimms.Controller import TopNController
from vimms.Environment import Environment

from vimms_gym.evaluation import evaluate, run_method
from vimms_gym.common import METHOD_TOPN, METHOD_RANDOM
from vimms_gym.experiments import preset_qcb_medium

env_alpha = 0.00
env_beta = 0.00
extract = True
params, max_peaks = preset_qcb_medium(None, alpha=env_alpha, beta=env_beta,
                                      extract_chromatograms=extract)

# %%
env_name = 'DDAEnv'
intensity_threshold = 0.5
max_peaks = 20

methods = [
    METHOD_RANDOM,
    METHOD_TOPN,
]
valid_random = True
n_eval_episodes = 1

# topN parameters
topN_N = 10
topN_rt_tol = 5
min_ms1_intensity = 5000

## Generate chemical sets for evaluation
fname = 'QCB_chems_medium.p'
chem_list = load_obj(fname)
chem_list = chem_list[0:n_eval_episodes]

# Evaluation

set_log_level_warning()
horizon = 1
out_dir = 'debug_TopN'

method_eval_results = {}
for method in methods:

    N = 0
    copy_params = copy.deepcopy(params)
    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }

    model = None
    if method == METHOD_TOPN:
        N = topN_N
        effective_rt_tol = topN_rt_tol
        copy_params = dict(params)
        copy_params['env']['use_dew'] = True
        copy_params['env']['rt_tol'] = effective_rt_tol

    banner = 'method = %s max_peaks = %d N = %d rt_tol = %d' % (method, max_peaks, N,
                                                                copy_params['env']['rt_tol'])
    print(banner)

    episodic_results = run_method(env_name, copy_params, max_peaks, chem_list, method, out_dir,
                                  N=N, min_ms1_intensity=min_ms1_intensity, model=model,
                                  print_eval=True, print_reward=True,
                                  intensity_threshold=intensity_threshold,
                                  mzml_prefix=method, horizon=horizon, valid_random=valid_random)
    eval_results = [er.eval_res for er in episodic_results]
    method_eval_results[method] = eval_results
    print()

#### Test classic controllers in ViMMS

enable_spike_noise = params['noise']['enable_spike_noise']
ionisation_mode = params['env']['ionisation_mode']
isolation_window = params['env']['isolation_window']
mz_tol = params['env']['mz_tol']
rt_range = params['chemical_creator']['rt_range']
method = 'TopN_Controller'
print('method = %s' % method)
print()

effective_rt_tol = topN_rt_tol
effective_N = topN_N
eval_results = []
for i in range(len(chem_list)):

    spike_noise = None
    if enable_spike_noise:
        noise_params = params['noise']
        noise_density = noise_params['noise_density']
        noise_max_val = noise_params['noise_max_val']
        noise_min_mz = noise_params['mz_range'][0]
        noise_max_mz = noise_params['mz_range'][1]
        spike_noise = UniformSpikeNoise(noise_density, noise_max_val, min_mz=noise_min_mz,
                                        max_mz=noise_max_mz)

    # print(effective_N, mz_tol, effective_rt_tol, min_ms1_intensity)
    chems = chem_list[i]
    mass_spec = IndependentMassSpectrometer(ionisation_mode, chems, spike_noise=spike_noise)
    controller = TopNController(ionisation_mode, effective_N, isolation_window, mz_tol,
                                effective_rt_tol,
                                min_ms1_intensity)
    env = Environment(mass_spec, controller, rt_range[0], rt_range[1], progress_bar=False,
                      out_dir=out_dir,
                      out_file='%s_%d.mzML' % (method, i), save_eval=True)
    env.run()

    eval_res = evaluate(env, intensity_threshold)
    # eval_res['total_rewards'] = 0
    eval_results.append(eval_res)
    print('Episode %d finished' % i)
    print(eval_res)
    print()

method_eval_results[method] = eval_results

data = []
for method in method_eval_results:
    eval_results = method_eval_results[method]
    for eval_res in eval_results:
        try:
            total_rewards = float(eval_res['total_rewards'])
        except KeyError:
            total_rewards = 0.0

        try:
            invalid_action_count = float(eval_res['invalid_action_count'])
        except KeyError:
            invalid_action_count = 0.0

        row = (
            method,
            total_rewards,
            invalid_action_count,
            float(eval_res['coverage_prop']),
            float(eval_res['intensity_prop']),
            float(eval_res['ms1ms2_ratio']),
            float(eval_res['efficiency']),
            float(eval_res['precision']),
            float(eval_res['recall']),
            float(eval_res['f1']),
        )
        data.append(row)

df = pd.DataFrame(data, columns=['method', 'total_rewards', 'invalid_action_count',
                                 'coverage_prop', 'intensity_prop', 'ms1/ms2_ratio', 'efficiency',
                                 'precision', 'recall', 'f1'])

# set the display option to show all columns
pd.set_option('display.max_columns', None)
print(df)