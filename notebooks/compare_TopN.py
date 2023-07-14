import random as rand

import numpy as np
from vimms.ChemicalSamplers import UniformRTAndIntensitySampler, \
    ConstantChromatogramSampler, FormulaSampler, EvenMZFormulaSampler
from vimms.Chemicals import ChemicalMixtureCreator
from vimms.Common import POSITIVE, DummyFormula
from vimms.Controller import TopNController
from vimms.Environment import Environment
from vimms.MassSpec import IndependentMassSpectrometer

from vimms_gym.env import DDAEnv
from vimms_gym.evaluation import Episode
from vimms_gym.policy import topN_policy
from vimms_gym.wrappers import flatten_dict_observations

np.random.seed(0)
rand.seed(0)

store_obs = True
print_reward = True
max_peaks = 30  # TODO
N = 10
min_ms1_intensity = 5000
intensity_threshold = 0.5

mz_range = (0, 300)
rt_range = (100, 102)
n_chemicals = 100

em = EvenMZFormulaSampler()
em.step = 1
ri = UniformRTAndIntensitySampler(min_rt=rt_range[0], max_rt=rt_range[1])
cs = ConstantChromatogramSampler()
cm = ChemicalMixtureCreator(em, rt_and_intensity_sampler=ri, chromatogram_sampler=cs)
chems = cm.sample(n_chemicals, 2)

env_params = {
    'chemical_creator': {
        'n_chemicals': n_chemicals,
        'mz_sampler': em,
        'ri_sampler': ri,
        'cr_sampler': cs,
    },
    'noise': {
        'enable_spike_noise': False,
        'noise_density': 0.1,
        'noise_max_val': 1E3,
        'mz_range': mz_range
    },
    'env': {
        'ionisation_mode': POSITIVE,
        'rt_range': rt_range,
        'isolation_window': 0.7,
        'use_dew': True,
        'mz_tol': 10,
        'rt_tol': 5,
        'min_ms1_intensity': min_ms1_intensity
    }
}

env = DDAEnv(max_peaks, env_params)
env = flatten_dict_observations(env)
obs, info = env.reset(options={'chems': chems})
states = None
done = False

# lists to store episodic results
episode = Episode(obs)
episode_starts = np.ones((1,), dtype=bool)
while not done:  # repeat until episode is done

    unwrapped_obs = env.env.state  # access the state attribute in DDAEnv
    action_masks = env.env.action_masks()

    # select an action depending on the observation and method
    features = env.features
    action = topN_policy(unwrapped_obs, features, N, min_ms1_intensity)
    action_probs = []

    # print(action, unwrapped_obs)

    # make one step through the simulation
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated
    episode_starts = done

    # store new episodic information
    if obs is not None:
        obs_to_store = obs if store_obs else None
        episode.add_step_data(action, action_probs, obs_to_store, reward, info)

    if print_reward and episode.num_steps % 500 == 0:
        print('steps\t', episode.num_steps, '\ttotal rewards\t',
              episode.get_total_rewards())

    # if episode is finished, break
    if done:
        break

# save mzML and other info useful for evaluation of the ViMMS environment
out_file = 'ddaenv_out.mzML'
env.write_mzML('results', out_file)

# environment will be evaluated here
eval_res = episode.evaluate_environment(env, intensity_threshold)
print(eval_res)

##### now the same chemicals, but run it through standard ViMMS environment

mass_spec = IndependentMassSpectrometer(POSITIVE, chems, spike_noise=None)
controller = TopNController(POSITIVE, N, 0.7, 10,
                            5,
                            min_ms1_intensity)
env = Environment(mass_spec, controller, rt_range[0], rt_range[1], progress_bar=False,
                  out_dir='results',
                  out_file='vimmsenv_out.mzML', save_eval=True)
env.run()
