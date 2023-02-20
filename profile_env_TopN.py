import os

from stable_baselines3.common.env_checker import check_env
from vimms.Common import load_obj

from experiments import preset_qcb_small, ENV_QCB_SMALL_GAUSSIAN, ENV_QCB_MEDIUM_GAUSSIAN, \
    ENV_QCB_LARGE_GAUSSIAN, ENV_QCB_SMALL_EXTRACTED, ENV_QCB_MEDIUM_EXTRACTED, \
    ENV_QCB_LARGE_EXTRACTED, preset_qcb_medium, preset_qcb_large
from vimms_gym.common import METHOD_TOPN, EVAL_F1_INTENSITY_THRESHOLD
from vimms_gym.env import DDAEnv
from vimms_gym.evaluation import run_method

import numpy as np
np.random.seed(0)

preset = "QCB_resimulated_medium"
model_name = "DQN"
alpha = 0.25
beta = 0.25

# choose one preset and generate parameters for it
presets = {
    ENV_QCB_SMALL_GAUSSIAN: {'f': preset_qcb_small, 'extract': False},
    ENV_QCB_MEDIUM_GAUSSIAN: {'f': preset_qcb_medium, 'extract': False},
    ENV_QCB_LARGE_GAUSSIAN: {'f': preset_qcb_large, 'extract': False},
    ENV_QCB_SMALL_EXTRACTED: {'f': preset_qcb_small, 'extract': True},
    ENV_QCB_MEDIUM_EXTRACTED: {'f': preset_qcb_medium, 'extract': True},
    ENV_QCB_LARGE_EXTRACTED: {'f': preset_qcb_large, 'extract': True},
}
preset_func = presets[preset]['f']
extract = presets[preset]['extract']
params, max_peaks = preset_func(model_name, alpha=alpha, beta=beta,
                                extract_chromatograms=extract)

print(params)
print(max_peaks)

env = DDAEnv(max_peaks, params)
check_env(env)

fname = os.path.join('notebooks', 'QCB_resimulated_medium', 'QCB_chems_medium.p')
chem_list = load_obj(fname)
chem_list = [chem_list[0]]
method = METHOD_TOPN
out_dir = 'profile'
HISTORY_HORIZON = 4

results = run_method(None, params, max_peaks, chem_list, method, out_dir,
           N=10, min_ms1_intensity=5000, model=None,
           print_eval=True, print_reward=True, mzml_prefix=None,
           intensity_threshold=EVAL_F1_INTENSITY_THRESHOLD, horizon=HISTORY_HORIZON,
           write_mzML=False)

num_ms1_scans = results[0].eval_res['num_ms1_scans']
num_ms2_scans = results[0].eval_res['num_ms2_scans']
print(num_ms1_scans, num_ms2_scans)
assert(num_ms1_scans == 278)
assert(num_ms2_scans == 1444)