from abc import abstractmethod
from collections import defaultdict
from random import randrange

import gym
import numpy as np
import pylab as plt
from gym import spaces
from loguru import logger
from vimms.ChemicalSamplers import UniformMZFormulaSampler, UniformRTAndIntensitySampler
from vimms.Chemicals import ChemicalMixtureCreator
from vimms.Common import set_log_level_warning
from vimms.Controller import AgentBasedController
from vimms.Environment import Environment
from vimms.MassSpec import IndependentMassSpectrometer

from vimms_gym.agents import DataDependantAcquisitionAgent, DataDependantAction
from vimms_gym.chemicals import generate_chemicals
from vimms_gym.features import CleanerTopNExclusion, Feature

REPEATED_MS1_REWARD = -0.1
REPEATED_FRAG_REWARD = -10
INVALID_MOVE_REWARD = -1

class DDAEnv(gym.Env):
    """
    Wrapper ViMMS Environment that follows gym interface
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, max_peaks, params):
        super().__init__()
        assert len(params) > 0
        self.max_peaks = max_peaks
        self.in_dim = self.max_peaks + 1  # 0 is for MS1
        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()
        self.features = []
        self.step_no = 0

        self.chemical_creator_params = params['chemical_creator']
        self.noise_params = params['noise']
        self.env_params = params['env']
        self.repeated_ms1_reward = params['ms1_reward'] \
            if 'ms1_reward' in params else REPEATED_MS1_REWARD
        self.invalid_move_reward = params['invalid_move_reward'] \
            if 'invalid_move_reward' in params else INVALID_MOVE_REWARD
        self.repeated_frag_reward = params['repeated_frag_reward'] \
            if 'repeated_frag_reward' in params else REPEATED_FRAG_REWARD

        self.mz_tol = self.env_params['mz_tol']
        self.rt_tol = self.env_params['rt_tol']
        self.min_ms1_intensity = self.env_params['min_ms1_intensity']
        self.isolation_window = self.env_params['isolation_window']

        self.mass_spec = None
        self.controller = None
        self.vimms_env = None
        self._initial_values()

    def _get_action_space(self):
        """
        Defines action space
        """
        return spaces.Discrete(self.in_dim)

    def _get_observation_space(self):
        """
        Defines observation spaces of m/z, RT and intensity values
        """
        combined_spaces = spaces.Dict({
            'intensities': spaces.Box(low=0, high=1, shape=(self.max_peaks,)),
            'ms_level': spaces.Box(low=1, high=2, shape=(1,)),    
            'fragmented': spaces.MultiBinary(self.max_peaks),
            'excluded': spaces.MultiBinary(self.max_peaks),
            'above_min_intensity': spaces.MultiBinary(self.max_peaks),
            'valid_actions': spaces.MultiBinary(self.in_dim),
            'fragmented_above': spaces.Box(low=0, high=self.max_peaks, shape=(1,)),
            'fragmented_below': spaces.Box(low=0, high=self.max_peaks, shape=(1,)),
            'unfragmented_above': spaces.Box(low=0, high=self.max_peaks, shape=(1,)),
            'unfragmented_below': spaces.Box(low=0, high=self.max_peaks, shape=(1,)),
            'excluded_above': spaces.Box(low=0, high=self.max_peaks, shape=(1,)),
            'excluded_below': spaces.Box(low=0, high=self.max_peaks, shape=(1,)),
            'fragmented_excluded_above': spaces.Box(low=0, high=self.max_peaks, shape=(1,)),
            'fragmented_excluded_below': spaces.Box(low=0, high=self.max_peaks, shape=(1,)),
            'elapsed_scans_since_start': spaces.Box(low=0, high=np.inf, shape=(1,)),
            'elapsed_scans_since_last_ms1': spaces.Box(low=0, high=np.inf, shape=(1,)),
        })
        return combined_spaces

    def _initial_state(self):
        features = {
            'intensities': np.zeros(self.max_peaks),
            'ms_level': np.zeros(1),
            'fragmented': np.zeros(self.max_peaks),
            'excluded': np.zeros(self.max_peaks),
            'above_min_intensity': np.zeros(self.max_peaks),
            'valid_actions': np.zeros(self.in_dim),
            'fragmented_above': np.zeros(1),
            'fragmented_below': np.zeros(1),
            'unfragmented_above': np.zeros(1),
            'unfragmented_below': np.zeros(1),
            'excluded_above': np.zeros(1),
            'excluded_below': np.zeros(1),
            'fragmented_excluded_above': np.zeros(1),
            'fragmented_excluded_below': np.zeros(1),
            'elapsed_scans_since_start': np.zeros(1),
            'elapsed_scans_since_last_ms1': np.zeros(1)
        }
        return features

    def _get_state(self, scan_to_process, dda_action):
        """
        Converts a scan to a state
        """
        state = self._scan_to_state(dda_action, scan_to_process)
        self._update_counts(state)
        return state

    def _scan_to_state(self, dda_action, scan_to_process):
        # TODO: can be moved into its own class?

        self.elapsed_scans_since_start += 1
        if dda_action.ms_level == 1:

            # new ms1 scan, so initialise a new state
            mzs, rt, intensities = self._get_mzs_rt_intensities(scan_to_process)
            max_intensity = 0
            if len(intensities) > 0:
                max_intensity = max(intensities)

            # get the N most intense features first
            features = []
            sorted_indices = np.flip(intensities.argsort())
            for i in sorted_indices[0:self.max_peaks]:
                mz = mzs[i]
                original_intensity = intensities[i]
                if max_intensity > 0:
                    scaled_intensity = original_intensity / max_intensity
                fragmented = 0  # initially nothing has been fragmented
                is_excluded, weight = self.exclusion.is_excluded(mz, rt)
                excluded = 1 if is_excluded else 0
                above_min_intensity = 1 if original_intensity > self.min_ms1_intensity else 0
                feature = Feature(mz, rt, original_intensity, scaled_intensity, 
                fragmented, excluded, above_min_intensity)
                features.append(feature)
            self.features = features

            # convert features to state
            assert len(features) <= self.max_peaks
            state = self._initial_state()
            for i in range(len(features)):
                f = features[i]
                state['intensities'][i] = f.scaled_intensity
                state['fragmented'][i] = f.fragmented
                state['excluded'][i] = f.excluded
                state['above_min_intensity'][i] = f.above_min_intensity
                state['valid_actions'][i] = 1 # fragmentable

            state['ms_level'][0] = 1
            self.elapsed_scans_since_last_ms1 = 0

        elif dda_action.ms_level == 2:

            # same ms1 scan, update state
            state = self.state
            idx = dda_action.idx
            assert idx is not None

            # update fragmented count
            state['fragmented'][idx] += 1

            # update exclusion
            state['excluded'][idx] = 1
            self.exclusion.update(dda_action.mz, dda_action.rt)

            state['ms_level'][0] = 2
            self.elapsed_scans_since_last_ms1 += 1

        state['valid_actions'][-1] = 1  # ms1 action is always valid
        return state

    def _update_counts(self, state):

        # count fragmented above min intensity
        fragmented_above = np.sum((state['fragmented'] == 1) & (state['above_min_intensity'] == 1))

        # count fragmented below min intensity
        fragmented_below = np.sum((state['fragmented'] == 1) & (state['above_min_intensity'] == 0))

        # count unfragmented above min intensity
        unfragmented_above = np.sum(
            (state['fragmented'] == 0) & (state['above_min_intensity'] == 1))

        # count unfragmented below min intensity
        unfragmented_below = np.sum(
            (state['fragmented'] == 0) & (state['above_min_intensity'] == 0))

        # count excluded above min intensity
        excluded_above = np.sum((state['excluded'] == 1) & (state['above_min_intensity'] == 1))

        # count excluded below min intensity
        excluded_below = np.sum((state['excluded'] == 1) & (state['above_min_intensity'] == 0))

        # count fragmented and excluded above min intensity
        fragmented_excluded_above = np.sum(
            (state['fragmented'] == 1) & (state['excluded'] == 1) & (
                    state['above_min_intensity'] == 1))

        # count fragmented and excluded below min intensity
        fragmented_excluded_below = np.sum(
            (state['fragmented'] == 1) & (state['excluded'] == 1) & (
                    state['above_min_intensity'] == 0))

        state['fragmented_above'][0] = fragmented_above
        state['fragmented_below'][0] = fragmented_below
        state['unfragmented_above'][0] = unfragmented_above
        state['unfragmented_below'][0] = unfragmented_below
        state['excluded_above'][0] = excluded_above
        state['excluded_below'][0] = excluded_below
        state['fragmented_excluded_above'][0] = fragmented_excluded_above
        state['fragmented_excluded_below'][0] = fragmented_excluded_below
        state['elapsed_scans_since_start'][0] = self.elapsed_scans_since_start
        state['elapsed_scans_since_last_ms1'][0] = self.elapsed_scans_since_last_ms1

    def _initial_values(self):
        """
        Sets some initial values
        """
        if self.mass_spec is not None:
            self.mass_spec.fire_event(IndependentMassSpectrometer.ACQUISITION_STREAM_CLOSED)
            self.mass_spec.close()

        if self.vimms_env is not None:
            self.vimms_env.close_progress_bar()

        self.chems = []
        self.current_scan = None
        self.state = self._initial_state()
        self.episode_done = False
        self.last_ms1_scan = None
        self.last_reward = 0.0

        self.elapsed_scans_since_start = 0
        self.elapsed_scans_since_last_ms1 = 0

        # track excluded ions
        self.exclusion = CleanerTopNExclusion(self.mz_tol, self.rt_tol)

        # track fragmented chemicals
        self.frag_chem_intensity = defaultdict(float)

        # needed for SubprocVecEnv
        set_log_level_warning()

    def step(self, action):
        """
        Execute one time step within the environment
        One step = a block of MS2 scans + 1 MS1 scan
        """
        self.step_no += 1
        info = {}

        # get next scan and next state
        next_scan, episode_done, dda_action, is_valid = self._one_step(
            action, self.current_scan)
        self.episode_done = episode_done

        if not episode_done:
            assert next_scan.ms_level == dda_action.ms_level
            self.state = self._get_state(next_scan, dda_action)
            self.last_reward = self._compute_reward(next_scan, dda_action, is_valid)
            if next_scan.ms_level == 1:
                self.last_ms1_scan = next_scan
            self.current_scan = next_scan
        else:
            self.state = None
            self.last_reward = 0
            self.current_scan = None

        return self.state, self.last_reward, self.episode_done, info

    def _one_step(self, action, scan_to_process):
        """
        Advance the simulation by processing one scan
        """

        # Take action by instructing the DDA agent to perform MS1 scan or
        # target a particular ion for MS2 fragmentation
        is_valid = True
        if action == self.max_peaks:
            dda_action = self.controller.agent.target_ms1()
        else:
            # 0 .. N-1 is the index of precursor ion to fragment
            idx = action
            try:
                f = self.features[idx]
                target_mz = f.mz
                target_rt = f.rt
                target_original_intensity = f.original_intensity
                target_scaled_intensity = f.scaled_intensity
            except IndexError:
                target_mz = 0
                target_rt = 0
                target_original_intensity = 0
                target_scaled_intensity = 0

            # check if an invalid fragmentation action has been selected
            # if yes, give negative reward (later) and advance the state of simulation by
            # doing an ms1 scan
            dda_action = self.controller.agent.target_ms2(target_mz, target_rt,
                                                          target_original_intensity, 
                                                          target_scaled_intensity, idx)
            if not dda_action.valid:
                is_valid = False

                # this makes learning much harder!
                # dda_action = self.controller.agent.target_ms1()

        # Ask controller to process the scan based on action
        # Advance mass spec to process the resulting scan, and check if we're done.
        self.mass_spec.dispatch_scan(scan_to_process)
        self.controller.update_state_after_scan(scan_to_process)
        self.vimms_env._update_progress_bar(scan_to_process)

        # Check if we're done
        episode_done = True if self.mass_spec.time > self.vimms_env.max_time else False
        if episode_done:
            return None, episode_done, dda_action, is_valid
        next_processed_scan_id = self.controller.next_processed_scan_id

        # Generate new scan but don't call the controller yet
        new_scan = self.mass_spec.step(call_controller=False)
        assert new_scan is not None
        assert new_scan.scan_id == next_processed_scan_id
        next_scan_to_process = new_scan
        return next_scan_to_process, episode_done, dda_action, is_valid

    @abstractmethod
    def _compute_reward(self, next_scan, dda_action, is_valid):
        """
        Give a constant reward for MS1 scan
        Compute MS2 reward by summing the total fragmented precursor intensities.
        """
        frag_event = next_scan.fragevent
        reward = 0

        # if not a valid move, give a large negative reward
        if not is_valid:
            reward = self.invalid_move_reward
        else:

            # if ms1, give constant positive reward
            # if invalid move, give constant negative reward
            if dda_action.ms_level == 1:

                # if ms2 and schedule ms1 ...
                if self.current_scan.ms_level == 2:
                    # give reward proportional to the number of unexcluded precursors in the ms1 scan
                    excluded_count = float(self.state['excluded_above'] + self.state['excluded_below'])
                    reward = (self.max_peaks - excluded_count) / self.max_peaks
                else:
                    # repeated scheduling of MS1 is not desirable
                    reward = self.repeated_ms1_reward

            # if ms2, give fragmented chemical intensity as the reward
            elif dda_action.ms_level == 2:
                if frag_event is not None:  # something has been fragmented

                    # look up previous fragmented intensity for this chem
                    chem = frag_event.chem
                    prev_intensity = self.frag_chem_intensity[chem]

                    # compute the current fragmented intensity for this chem
                    new_intensity = np.log(np.sum(frag_event.parents_intensity))
                    self.frag_chem_intensity[chem] = new_intensity

                    # calculate difference between successive fragmentations of the same chem
                    # intensity_diff = new_intensity - prev_intensity
                    # reward = intensity_diff

                    # alternative thresholded reward scheme
                    reward = self.repeated_frag_reward
                    intensity_diff = new_intensity - prev_intensity
                    if intensity_diff > 1:
                        reward = intensity_diff
                        self.frag_chem_intensity[chem] = new_intensity
                    reward = self._clip_reward(reward, 10)

        assert -1 <= reward <= 1
        return reward

    def _clip_reward(self, reward, max_reward):
        # clip reward to [-1, 1]
        reward = min(reward, max_reward) if reward >= 0 else max(reward, -max_reward)
        reward = reward / max_reward
        return reward

    def reset(self, chems=None):
        """
        Reset the state of the environment to an initial state
        """
        # 1. Reset initial states
        self._initial_values()

        # 2. Reset generated chemicals
        self.chems = generate_chemicals(self.chemical_creator_params) if chems is None else chems

        # 3. Reset ViMMS environment
        self.mass_spec = self._reset_mass_spec(self.chems, self.env_params, self.noise_params)
        self.controller = self._reset_controller(self.env_params)
        self.vimms_env = self._reset_vimms_environment(self.mass_spec, self.controller,
                                                       self.env_params)

        # 4. Generate the initial scan in vimms environment
        self.vimms_env._set_initial_values()
        self.mass_spec.register_event(IndependentMassSpectrometer.MS_SCAN_ARRIVED,
                                      self.vimms_env.add_scan)
        self.mass_spec.register_event(IndependentMassSpectrometer.ACQUISITION_STREAM_OPENING,
                                      self._handle_acquisition_open)
        self.mass_spec.register_event(IndependentMassSpectrometer.ACQUISITION_STREAM_CLOSED,
                                      self.vimms_env.handle_acquisition_closing)
        self.mass_spec.register_event(IndependentMassSpectrometer.STATE_CHANGED,
                                      self.vimms_env.handle_state_changed)

        # 5. Generate initial scan when the acquisition opens
        self.mass_spec.fire_event(  # call self._handle_acquisition_open() below
            IndependentMassSpectrometer.ACQUISITION_STREAM_OPENING)
        self.episode_done = False
        ms1_action = DataDependantAction()  # same as the default initial scan
        self.state = self._get_state(self.current_scan, ms1_action)
        return self.state

    def close(self):
        pass

    def _reset_mass_spec(self, chems, env_params, noise_params):
        """
        Generates new mass spec
        """
        # noise_density = noise_params['noise_density']
        # noise_max_val = noise_params['noise_max_val']
        # noise_min_mz = noise_params['mz_range'][0]
        # noise_max_mz = noise_params['mz_range'][1]
        # spike_noise = UniformSpikeNoise(noise_density, noise_max_val, min_mz=noise_min_mz,
        #                                 max_mz=noise_max_mz)
        spike_noise = None
        ionisation_mode = env_params['ionisation_mode']
        mass_spec = IndependentMassSpectrometer(ionisation_mode, chems, None,
                                                spike_noise=spike_noise)
        return mass_spec

    def _reset_controller(self, env_params):
        """
        Generates new controller
        """
        agent = DataDependantAcquisitionAgent(self.isolation_window)
        controller = AgentBasedController(agent)
        return controller

    def _reset_vimms_environment(self, mass_spec, controller, env_params):
        """
        Generates new ViMMS environment to run controller and mass spec together
        """
        min_rt = env_params['rt_range'][0]
        max_rt = env_params['rt_range'][1]
        vimms_env = Environment(mass_spec, controller, min_rt, max_rt, progress_bar=False)
        return vimms_env

    def _get_chemicals(self, chemical_creator_params):
        """
        Generates new set of chemicals
        """
        min_mz = chemical_creator_params['mz_range'][0]
        max_mz = chemical_creator_params['mz_range'][1]
        min_rt = chemical_creator_params['rt_range'][0]
        max_rt = chemical_creator_params['rt_range'][1]
        min_log_intensity = np.log(chemical_creator_params['intensity_range'][0])
        max_log_intensity = np.log(chemical_creator_params['intensity_range'][1])
        n_chemicals_range = chemical_creator_params['n_chemicals']
        if n_chemicals_range[0] == n_chemicals_range[1]:
            n_chems = n_chemicals_range[0]
        else:
            n_chems = randrange(n_chemicals_range[0], n_chemicals_range[1])

        if 'mz_sampler' in chemical_creator_params:
            mz_sampler = chemical_creator_params['mz_sampler']
        else:  # sample chemicals uniformly
            mz_sampler = UniformMZFormulaSampler(min_mz=min_mz, max_mz=max_mz)

        if 'rt_sampler' in chemical_creator_params:
            ri_sampler = chemical_creator_params['ri_sampler']
        else:
            ri_sampler = UniformRTAndIntensitySampler(min_rt=min_rt, max_rt=max_rt,
                                                      min_log_intensity=min_log_intensity,
                                                      max_log_intensity=max_log_intensity)

        cm = ChemicalMixtureCreator(mz_sampler, rt_and_intensity_sampler=ri_sampler)
        chems = cm.sample(n_chems, 2, include_adducts_isotopes=False)
        return chems

    def _handle_acquisition_open(self):
        """
        Open acquisition and generates an MS1 scan to process
        """
        logger.debug('Acquisition open')
        # send the initial custom scan to start the custom scan generation process
        params = self.controller.get_initial_scan_params()

        # unlike the normal Vimms environment, here we generate scan but don't call the controller
        # yet (time is also not incremented)
        self.current_scan = self.mass_spec.step(params=params, call_controller=False)

    def _get_mzs_rt_intensities(self, scan_to_process):
        """
        Extracts mzs, rt, and intensities values from a scan
        """
        mzs = scan_to_process.mzs
        intensities = scan_to_process.intensities
        rt = scan_to_process.rt
        assert mzs.shape == intensities.shape
        return mzs, rt, intensities

    def render(self, mode='human', close=False):
        """
        Render the environment to the screen
        """
        logger.info('step %d %s reward %f done %s' % (
            self.step_no, self.current_scan,
            self.last_reward, self.episode_done))

        self._plot_scan(self.current_scan)

    def _plot_scan(self, scan):
        """
        Plot a scan
        Args:
            scan: a [vimms.MassSpec.Scan][] object.

        Returns: None

        """
        plt.figure()
        for i in range(scan.num_peaks):
            x1 = scan.mzs[i]
            x2 = scan.mzs[i]
            y1 = 0
            y2 = np.log(scan.intensities[i])
            a = [[x1, y1], [x2, y2]]
            plt.plot(*zip(*a), marker='', color='r', ls='-', lw=1)
        plt.title('Scan {0} {1}s -- {2} peaks (ms_level {3})'.format(scan.scan_id, scan.rt,
                                                                     scan.num_peaks,
                                                                     scan.ms_level))
        plt.xlabel('m/z')
        plt.ylabel('log intensity')
        plt.show()

    def write_mzML(self, out_dir, out_file):
        self.vimms_env.write_mzML(out_dir, out_file)
