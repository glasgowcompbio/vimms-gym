from abc import abstractmethod
from copy import deepcopy

import gym
import numpy as np
import pylab as plt
from gym import spaces
from loguru import logger
from matplotlib.backends.backend_agg import FigureCanvasAgg
from vimms.Common import set_log_level_warning
from vimms.Controller import AgentBasedController
from vimms.Environment import Environment
from vimms.MassSpec import IndependentMassSpectrometer
from vimms.Noise import UniformSpikeNoise
from vimms.Roi import RoiBuilder, SmartRoiParams, RoiBuilderParams

from vimms_gym.agents import DataDependantAcquisitionAgent, DataDependantAction
from vimms_gym.chemicals import generate_chemicals
from vimms_gym.common import clip_value, INVALID_MOVE_REWARD, RENDER_HUMAN, RENDER_RGB_ARRAY, \
    render_scan, ALPHA, BETA, NO_FRAGMENTATION_REWARD, CLIPPED_INTENSITY_LOW, \
    CLIPPED_INTENSITY_HIGH, MAX_OBSERVED_LOG_INTENSITY, MS1_REWARD_SHAPE, SKIP_MS2_SPECTRA
from vimms_gym.env_utils import scale_intensities, update_feature_roi
from vimms_gym.features import CleanerTopNExclusion, Feature


class DDAEnv(gym.Env):
    """
    Wrapper ViMMS Environment that follows gym interface
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, max_peaks, params):
        super().__init__()
        assert len(params) > 0
        self.max_peaks = max_peaks
        self.in_dim = self.max_peaks + 1  # 0 is for MS1

        self.chemical_creator_params = params['chemical_creator']
        self.noise_params = params['noise']
        self.env_params = params['env']
        self.alpha = ALPHA if 'alpha' not in self.env_params else self.env_params['alpha']
        self.beta = BETA if 'beta' not in self.env_params else self.env_params['beta']

        self.use_dew = self.env_params['use_dew']
        self.mz_tol = self.env_params['mz_tol']
        self.rt_tol = self.env_params['rt_tol']
        self.min_ms1_intensity = self.env_params['min_ms1_intensity']
        self.isolation_window = self.env_params['isolation_window']

        try:
            self.roi_params = self.env_params['roi_params']
        except KeyError:
            self.roi_params = RoiBuilderParams()

        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()

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
        lo = CLIPPED_INTENSITY_LOW
        hi = CLIPPED_INTENSITY_HIGH

        spaces_dict = {
            # precursor ion features
            'intensities': spaces.Box(low=lo, high=hi, shape=(self.max_peaks,)),
            'fragmented': spaces.MultiBinary(self.max_peaks),
            'excluded': spaces.MultiBinary(self.max_peaks),

            # roi features
            'roi_length': spaces.Box(
                low=0, high=1, shape=(self.max_peaks,)),
            'roi_elapsed_time_since_last_frag': spaces.Box(
                low=0, high=1, shape=(self.max_peaks,)),
            'roi_intensity_at_last_frag': spaces.Box(
                low=lo, high=hi, shape=(self.max_peaks,)),
            'roi_min_intensity_since_last_frag': spaces.Box(
                low=lo, high=hi, shape=(self.max_peaks,)),
            'roi_max_intensity_since_last_frag': spaces.Box(
                low=lo, high=hi, shape=(self.max_peaks,)),

            # roi intensity features
            'roi_intensities_2': spaces.Box(low=lo, high=hi, shape=(self.max_peaks,)),
            'roi_intensities_3': spaces.Box(low=lo, high=hi, shape=(self.max_peaks,)),
            'roi_intensities_4': spaces.Box(low=lo, high=hi, shape=(self.max_peaks,)),
            'roi_intensities_5': spaces.Box(low=lo, high=hi, shape=(self.max_peaks,)),
            'roi_intensities_6': spaces.Box(low=lo, high=hi, shape=(self.max_peaks,)),
            'roi_intensities_7': spaces.Box(low=lo, high=hi, shape=(self.max_peaks,)),
            'roi_intensities_8': spaces.Box(low=lo, high=hi, shape=(self.max_peaks,)),
            'roi_intensities_9': spaces.Box(low=lo, high=hi, shape=(self.max_peaks,)),
            'avg_roi_intensities': spaces.Box(low=lo, high=hi, shape=(self.max_peaks,)),

            'roi_intensity_diff_1': spaces.Box(low=-1, high=1, shape=(self.max_peaks,)),
            'roi_intensity_diff_2': spaces.Box(low=-1, high=1, shape=(self.max_peaks,)),
            'roi_intensity_diff_3': spaces.Box(low=-1, high=1, shape=(self.max_peaks,)),
            'roi_intensity_diff_4': spaces.Box(low=-1, high=1, shape=(self.max_peaks,)),
            'roi_intensity_diff_5': spaces.Box(low=-1, high=1, shape=(self.max_peaks,)),
            'roi_intensity_diff_6': spaces.Box(low=-1, high=1, shape=(self.max_peaks,)),
            'roi_intensity_diff_7': spaces.Box(low=-1, high=1, shape=(self.max_peaks,)),
            'roi_intensity_diff_8': spaces.Box(low=-1, high=1, shape=(self.max_peaks,)),

            # valid action indicators, last action and current ms level
            'valid_actions': spaces.MultiBinary(self.in_dim),
            'last_action': spaces.Discrete(self.in_dim + 1),
            'ms_level': spaces.Discrete(2),  # either MS1 or MS2 scans

            # various other counts
            'fragmented_count': spaces.Box(low=0, high=1, shape=(1,)),
            'unfragmented_count': spaces.Box(low=0, high=1, shape=(1,)),
            'excluded_count': spaces.Box(low=0, high=1, shape=(1,)),
            'unexcluded_count': spaces.Box(low=0, high=1, shape=(1,)),
            'elapsed_scans_since_start': spaces.Box(low=0, high=1, shape=(1,)),
            'elapsed_scans_since_last_ms1': spaces.Box(low=0, high=1, shape=(1,)),
        }

        if not self.use_dew:
            del spaces_dict['excluded']
            del spaces_dict['excluded_count']
            del spaces_dict['unexcluded_count']

        combined_spaces = spaces.Dict(spaces_dict)
        return combined_spaces

    def _initial_state(self):
        features = {
            # precursor ion features
            'intensities': np.zeros(self.max_peaks, dtype=np.float32),
            'fragmented': np.zeros(self.max_peaks, dtype=np.float32),
            'excluded': np.zeros(self.max_peaks, dtype=np.float32),

            # roi features
            'roi_length': np.zeros(self.max_peaks, dtype=np.float32),
            'roi_elapsed_time_since_last_frag': np.zeros(self.max_peaks, dtype=np.float32),
            'roi_intensity_at_last_frag': np.zeros(self.max_peaks, dtype=np.float32),
            'roi_min_intensity_since_last_frag': np.zeros(self.max_peaks, dtype=np.float32),
            'roi_max_intensity_since_last_frag': np.zeros(self.max_peaks, dtype=np.float32),

            # roi intensity features
            'roi_intensities_2': np.zeros(self.max_peaks, dtype=np.float32),
            'roi_intensities_3': np.zeros(self.max_peaks, dtype=np.float32),
            'roi_intensities_4': np.zeros(self.max_peaks, dtype=np.float32),
            'roi_intensities_5': np.zeros(self.max_peaks, dtype=np.float32),
            'roi_intensities_6': np.zeros(self.max_peaks, dtype=np.float32),
            'roi_intensities_7': np.zeros(self.max_peaks, dtype=np.float32),
            'roi_intensities_8': np.zeros(self.max_peaks, dtype=np.float32),
            'roi_intensities_9': np.zeros(self.max_peaks, dtype=np.float32),
            'avg_roi_intensities': np.zeros(self.max_peaks, dtype=np.float32),

            'roi_intensity_diff_1': np.zeros(self.max_peaks, dtype=np.float32),
            'roi_intensity_diff_2': np.zeros(self.max_peaks, dtype=np.float32),
            'roi_intensity_diff_3': np.zeros(self.max_peaks, dtype=np.float32),
            'roi_intensity_diff_4': np.zeros(self.max_peaks, dtype=np.float32),
            'roi_intensity_diff_5': np.zeros(self.max_peaks, dtype=np.float32),
            'roi_intensity_diff_6': np.zeros(self.max_peaks, dtype=np.float32),
            'roi_intensity_diff_7': np.zeros(self.max_peaks, dtype=np.float32),
            'roi_intensity_diff_8': np.zeros(self.max_peaks, dtype=np.float32),

            # valid action indicators
            'valid_actions': np.zeros(self.in_dim, dtype=np.float32),
            'ms_level': 0,
            'last_action': 0,

            # various other counts
            'fragmented_count': np.zeros(1, dtype=np.float32),
            'unfragmented_count': np.zeros(1, dtype=np.float32),
            'excluded_count': np.zeros(1, dtype=np.float32),
            'unexcluded_count': np.zeros(1, dtype=np.float32),
            'elapsed_scans_since_start': np.zeros(1, dtype=np.float32),
            'elapsed_scans_since_last_ms1': np.zeros(1, dtype=np.float32)
        }

        if not self.use_dew:
            del features['excluded']
            del features['excluded_count']
            del features['unexcluded_count']

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
            state = self._scan_to_state_ms1(scan_to_process)

        elif dda_action.ms_level == 2:
            state = self._scan_to_state_ms2(dda_action, scan_to_process)

        state['valid_actions'][-1] = 1  # ms1 action is always valid
        state['last_action'] = self.last_action
        return state

    def _scan_to_state_ms1(self, scan_to_process):

        # store last action
        self.last_action = self.in_dim

        # new ms1 scan, so initialise a new state
        mzs, rt, intensities = self._get_mzs_rt_intensities(scan_to_process)
        self.roi_builder.update_roi(scan_to_process)
        live_rois = self.roi_builder.live_roi

        # used to quickly match feature to a live ROI
        # key: last (mz, rt, intensity) of an ROI, value: the ROI object
        last_datum_to_roi = {roi.get_last_datum(): roi for roi in live_rois}

        # get the N most intense features first
        features = []
        sorted_indices = np.flip(intensities.argsort())
        for i in sorted_indices[0:self.max_peaks]:
            mz = mzs[i]
            original_intensity = intensities[i]

            # initially nothing has been fragmented
            fragmented = False

            # update exclusion elapsed time for all features
            excluded = False
            if self.use_dew:
                current_rt = scan_to_process.rt
                excluded = self._get_excluded(mz, current_rt)

            try:
                last_datum = (mz, rt, original_intensity)
                roi = last_datum_to_roi[last_datum]
            except KeyError:
                # This happens only for noise peaks, which as no associated chemical
                # TODO: I think we should handle this better?
                roi = None

                # print('Missing: %f %f %f' % last_datum)
                # for roi in self.roi_builder.live_roi:
                #     print('%s: %s' % (roi, roi.get_last_datum()))

            feature = Feature(mz, rt, original_intensity, None,
                              fragmented, excluded, roi)
            features.append(feature)
        self.features = features

        # convert features to state
        num_features = len(features)
        assert num_features <= self.max_peaks
        state = self._initial_state()
        for i in range(num_features):
            f = features[i]
            state['intensities'][i] = f.original_intensity
            state['fragmented'][i] = 0 if not f.fragmented else 1
            if self.use_dew:
                state['excluded'][i] = 0 if not f.excluded else 1
            state['valid_actions'][i] = 1  # fragmentable
            if f.original_intensity < self.min_ms1_intensity:
                state['valid_actions'][i] = 0  # except when it's below min ms1 intensity
            update_feature_roi(f, i, state)  # update ROI information for this feature

        state['roi_intensity_at_last_frag'] = scale_intensities(
            state['roi_intensity_at_last_frag'], num_features, MAX_OBSERVED_LOG_INTENSITY)
        state['roi_min_intensity_since_last_frag'] = scale_intensities(
            state['roi_min_intensity_since_last_frag'], num_features, MAX_OBSERVED_LOG_INTENSITY)
        state['roi_max_intensity_since_last_frag'] = scale_intensities(
            state['roi_max_intensity_since_last_frag'], num_features, MAX_OBSERVED_LOG_INTENSITY)

        state['avg_roi_intensities'] = scale_intensities(
            state['avg_roi_intensities'], num_features, MAX_OBSERVED_LOG_INTENSITY)

        state['roi_intensity_diff_1'] = scale_intensities(
            state['roi_intensity_diff_1'], num_features, MAX_OBSERVED_LOG_INTENSITY)
        state['roi_intensity_diff_2'] = scale_intensities(
            state['roi_intensity_diff_2'], num_features, MAX_OBSERVED_LOG_INTENSITY)
        state['roi_intensity_diff_3'] = scale_intensities(
            state['roi_intensity_diff_3'], num_features, MAX_OBSERVED_LOG_INTENSITY)
        state['roi_intensity_diff_4'] = scale_intensities(
            state['roi_intensity_diff_4'], num_features, MAX_OBSERVED_LOG_INTENSITY)
        state['roi_intensity_diff_5'] = scale_intensities(
            state['roi_intensity_diff_5'], num_features, MAX_OBSERVED_LOG_INTENSITY)
        state['roi_intensity_diff_6'] = scale_intensities(
            state['roi_intensity_diff_6'], num_features, MAX_OBSERVED_LOG_INTENSITY)
        state['roi_intensity_diff_7'] = scale_intensities(
            state['roi_intensity_diff_7'], num_features, MAX_OBSERVED_LOG_INTENSITY)
        state['roi_intensity_diff_8'] = scale_intensities(
            state['roi_intensity_diff_8'], num_features, MAX_OBSERVED_LOG_INTENSITY)

        state['intensities'] = scale_intensities(
            state['intensities'], num_features, MAX_OBSERVED_LOG_INTENSITY)
        state['roi_intensities_2'] = scale_intensities(
            state['roi_intensities_2'], num_features, MAX_OBSERVED_LOG_INTENSITY)
        state['roi_intensities_3'] = scale_intensities(
            state['roi_intensities_3'], num_features, MAX_OBSERVED_LOG_INTENSITY)
        state['roi_intensities_4'] = scale_intensities(
            state['roi_intensities_4'], num_features, MAX_OBSERVED_LOG_INTENSITY)
        state['roi_intensities_5'] = scale_intensities(
            state['roi_intensities_5'], num_features, MAX_OBSERVED_LOG_INTENSITY)
        state['roi_intensities_6'] = scale_intensities(
            state['roi_intensities_6'], num_features, MAX_OBSERVED_LOG_INTENSITY)
        state['roi_intensities_7'] = scale_intensities(
            state['roi_intensities_7'], num_features, MAX_OBSERVED_LOG_INTENSITY)
        state['roi_intensities_8'] = scale_intensities(
            state['roi_intensities_8'], num_features, MAX_OBSERVED_LOG_INTENSITY)
        state['roi_intensities_9'] = scale_intensities(
            state['roi_intensities_9'], num_features, MAX_OBSERVED_LOG_INTENSITY)

        state['ms_level'] = 0
        self.elapsed_scans_since_last_ms1 = 0
        return state

    def _scan_to_state_ms2(self, dda_action, scan_to_process):

        # same ms1 scan, update state
        state = deepcopy(self.state)
        idx = dda_action.idx
        assert idx is not None

        # store last action
        self.last_action = idx

        # update fragmented flag
        state['fragmented'][idx] = 1

        # it's no longer valid to fragment this peak again
        state['valid_actions'][idx] = 0

        # find the feature that has been fragmented in this MS2 scan
        current_rt = scan_to_process.rt
        try:
            f = self.features[idx]
            f.fragmented = True

            # update exclusion for the selected feature
            if self.use_dew:
                self.exclusion.update(f.mz, f.rt)

            # set the ROI linked to this feature to be fragmented
            if f.roi is not None:  # FIXME: it shouldn't happen?
                f.roi.fragmented(current_rt)

        except IndexError:  # idx selects a non-existing feature
            pass

        # update exclusion elapsed time for all features
        if self.use_dew:
            for i in range(len(self.features)):
                f = self.features[i]
                excluded = self._get_excluded(f.mz, current_rt)
                state['excluded'][i] = excluded
                f.excluded = excluded
        state['ms_level'] = 1
        self.elapsed_scans_since_last_ms1 += 1
        return state

    def _get_excluded(self, mz, current_rt):
        # check if any boxes containing this m/z and RT
        return self.exclusion.exclusion_list.is_in_box(mz, current_rt)

    def _update_counts(self, state):

        num_features = len(self.features)
        fragmented = state['fragmented']

        # count fragmented
        fragmented_count = np.count_nonzero(fragmented[:num_features] > 0)

        # count unfragmented
        unfragmented_count = np.count_nonzero(fragmented[:num_features] == 0)

        state['fragmented_count'][0] = clip_value(fragmented_count, num_features)
        state['unfragmented_count'][0] = clip_value(unfragmented_count, num_features)
        assert (fragmented_count + unfragmented_count) == num_features

        state['elapsed_scans_since_start'][0] = clip_value(
            self.elapsed_scans_since_start, 10000)
        state['elapsed_scans_since_last_ms1'][0] = clip_value(
            self.elapsed_scans_since_last_ms1, 100)

        if self.use_dew:
            # count excluded and not-excluded
            excluded = state['excluded']
            excluded_count = np.count_nonzero(excluded[:num_features] > 0)
            unexcluded_count = np.count_nonzero(excluded[:num_features] == 0)

            state['excluded_count'][0] = clip_value(excluded_count, num_features)
            state['unexcluded_count'][0] = clip_value(unexcluded_count, num_features)
            assert (excluded_count + unexcluded_count) == num_features

    def _initial_values(self):
        """
        Sets some initial values
        """
        if self.mass_spec is not None:
            self.mass_spec.fire_event(IndependentMassSpectrometer.ACQUISITION_STREAM_CLOSED)
            self.mass_spec.close()

        if self.vimms_env is not None:
            self.vimms_env.close_progress_bar()

        self.features = []
        self.step_no = 0
        self.chems = []
        self.current_scan = None
        self.state = self._initial_state()
        self.episode_done = False
        self.last_ms1_scan = None
        self.last_reward = 0.0
        self.last_action = None

        self.elapsed_scans_since_start = 0
        self.elapsed_scans_since_last_ms1 = 0
        self.ms1_count = 0
        self.ms2_count = 0
        self.invalid_action_count = 0
        self.num_fragmented = 0

        # track regions of interest
        smartroi_params = SmartRoiParams()
        self.roi_builder = RoiBuilder(self.roi_params, smartroi_params=smartroi_params)

        # track excluded ions
        if self.use_dew:
            self.exclusion = CleanerTopNExclusion(self.mz_tol, self.rt_tol)

        # track fragmented chemicals
        self.frag_chem_intensity = {}
        self.frag_chem_time = {}

        # needed for SubprocVecEnv
        set_log_level_warning()

    def step(self, action):
        """
        Execute one time step within the environment
        One step = perform either an MS1 or an MS2 scan
        """
        self.step_no += 1
        info = {
            'current_scan_id': self.current_scan.scan_id,
        }

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
                self.ms1_count += 1
            else:
                self.ms2_count += 1
            self.current_scan = next_scan
        else:
            # TODO: could break wrappers .. leave the state unchanged when done?
            # self.state = None 
            self.current_scan = None
            self.last_reward = 0

            # penalise if no MS2 events have been performed
            if self.ms1_count > 0 and self.ms2_count == 0:
                self.last_reward = NO_FRAGMENTATION_REWARD

        return self.state, self.last_reward, self.episode_done, info

    def action_masks(self):
        mask = self.state['valid_actions'].astype(bool)
        return mask

    def _one_step(self, action, scan_to_process):
        """
        Advance the simulation by processing one scan
        """

        # Take action by instructing the DDA agent to perform MS1 scan or
        # target a particular ion for MS2 fragmentation
        is_valid = True
        if action == self.max_peaks:
            # ms1 action is always valid
            dda_action = self.controller.agent.target_ms1()
        else:
            # 0 .. N-1 is the index of precursor ion to fragment
            idx = action

            # check if targeting a feature that doesn't exist
            target_mz = 0
            target_rt = 0
            target_original_intensity = 0
            target_scaled_intensity = 0
            try:
                f = self.features[idx]
                if f.fragmented:
                    # check if targeting a feature that has been fragmented before
                    is_valid = False
                elif f.original_intensity < self.min_ms1_intensity:
                    # check if targeting a feature below min intensity
                    is_valid = False
                else:
                    # valid MS2 target
                    target_mz = f.mz
                    target_rt = f.rt
                    target_original_intensity = f.original_intensity
                    target_scaled_intensity = f.scaled_intensity

            except IndexError:
                is_valid = False

            dda_action = self.controller.agent.target_ms2(target_mz, target_rt,
                                                          target_original_intensity,
                                                          target_scaled_intensity, idx)

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
        frag_events = next_scan.fragevent
        reward = 0

        # if not a valid move, give a large negative reward
        if not is_valid:
            reward = INVALID_MOVE_REWARD
            self.invalid_action_count += 1
        else:

            if dda_action.ms_level == 1:
                # compute ms1 reward
                num_fragmented = self.elapsed_scans_since_last_ms1
                num_total = len(self.features)
                reward = self._compute_ms1_reward(num_fragmented, num_total, MS1_REWARD_SHAPE)

            elif dda_action.ms_level == 2:
                if frag_events is not None:  # some chemical has been fragmented

                    # TODO: assume only 1 chemical has been fragmented
                    # works for DDA but not for DIA
                    frag_event = frag_events[0]
                    frag_int = frag_event.parents_intensity[0]

                    # look up previous fragmented intensity for this chem
                    chem = frag_event.chem

                    # compute ms2 reward
                    reward = self._compute_ms2_reward(chem, frag_int, frag_event.query_rt)

                    # store new intensity and frag time into dictionaries
                    self.frag_chem_intensity[chem] = frag_int
                    self.frag_chem_time[chem] = frag_event.query_rt

                else:
                    # fragmenting a spike noise, or no chem associated with this, so we give no reward
                    reward = 0.0

        assert -1.0 <= reward <= 1
        return reward

    def _compute_ms1_reward(self, num_fragmented, num_total, alpha):
        """
        Calculate the MS1 reward as a function of the number of precursor ions
        that have been fragmented, using a decreasing exponential function that
        assigns more weight to the early precursor ions and gradually decreases
        the weight for the later ones.

        Args:
            num_fragmented (int): The number of precursor ions that have been
                fragmented so far. This should be a non-negative integer less than
                or equal to num_total.
            num_total (int): The total number of precursor ions in the current
                scan. This should be a positive integer.
            alpha (float): A scaling parameter that controls the shape of the
                reward function. A larger alpha value will result in a steeper
                curve that assigns more weight to the early precursor ions, while
                a smaller alpha value will result in a flatter curve that assigns
                more equal weight to all precursor ions. This should be a positive
                float.

        Returns:
            float: The MS1 reward for selecting an MS1 scan action, a value between
            0 and 1. The reward is based on the fraction of precursor ions that have
            been fragmented so far, using a decreasing exponential function that
            assigns more weight to the early precursor ions and gradually decreases
            the weight for the later ones.

        The MS1 reward function implemented here is based on the formula:

            reward = 1 - exp(-alpha * x)

        where x is the fraction of precursor ions that have been fragmented so far,
        calculated as num_fragmented / num_total. This formula corresponds to a
        decreasing exponential function that assigns more weight to the early
        precursor ions and gradually decreases the weight for the later ones.

        If alpha is set to 0, the MS1 reward will be a linear function of the number
        of precursor ions that have been fragmented, like the original reward
        function. If alpha is set to infinity, the MS1 reward will be 1 if no
        precursor ions have been fragmented and 0 if all precursor ions have been
        fragmented, effectively preventing the selection of MS1 scan actions
        altogether. A reasonable range for alpha depends on the specific application
        and can be determined through experimentation or domain expertise.
        """
        x = num_fragmented / num_total
        reward = 1 - np.exp(-alpha * x)
        return reward

    def _compute_ms2_reward(self, chem, frag_int, frag_time):

        # Apex reward
        rel_frag_time = frag_time - chem.rt
        chrom = chem.chromatogram
        apex_reward = self._compute_apex_reward(chrom, rel_frag_time)

        # intensity reward
        log_chem_frag_int = np.log(frag_int)
        normalized_log_intensity = log_chem_frag_int / MAX_OBSERVED_LOG_INTENSITY
        intensity_reward = np.clip(normalized_log_intensity, 0, 1)

        # Penalty for excessive fragmentation
        penalty = 0
        if chem in self.frag_chem_time:
            rel_last_frag_time = self.frag_chem_time[chem] - chem.rt
            last_apex_reward = self._compute_apex_reward(chrom, rel_last_frag_time)
            apex_reward_diff = apex_reward - last_apex_reward

            # When apex_reward improves upon the previous one, i.e. apex_reward_diff is positive,
            # the penalty is 0. Otherwise if apex_reward_diff is negative, then the penalty is
            # -apex_reward_diff, which is positive.
            penalty = max(0, -apex_reward_diff)

        # combined reward
        reward = apex_reward * intensity_reward * (1 - penalty)
        return reward

    def _compute_apex_reward(self, chrom, rel_frag_time):
        min_rt = chrom.min_rt
        max_rt = chrom.max_rt
        apex_time = chrom.get_apex_rt()
        normalized_frag_time = (rel_frag_time - min_rt) / (max_rt - min_rt)
        normalized_apex_time = (apex_time - min_rt) / (max_rt - min_rt)
        apex_distance = abs(normalized_frag_time - normalized_apex_time)
        apex_reward = 1 - apex_distance
        return apex_reward

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

        # check whether to enable spike noise
        enable_spike_noise = noise_params['enable_spike_noise']
        if enable_spike_noise:
            noise_density = noise_params['noise_density']
            noise_max_val = noise_params['noise_max_val']
            noise_min_mz = noise_params['mz_range'][0]
            noise_max_mz = noise_params['mz_range'][1]
            spike_noise = UniformSpikeNoise(noise_density, noise_max_val, min_mz=noise_min_mz,
                                            max_mz=noise_max_mz)
        else:
            spike_noise = None

        ionisation_mode = env_params['ionisation_mode']
        mass_spec = IndependentMassSpectrometer(ionisation_mode, chems, spike_noise=spike_noise,
                                                skip_ms2_spectra_generation=SKIP_MS2_SPECTRA)
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

    def render(self, mode='human'):
        """
        Render the environment to the screen
        """
        if mode == RENDER_RGB_ARRAY:
            # return RGB frame suitable for video
            fig = render_scan(self.current_scan)

            # FIXME: untested code!
            # https://stackoverflow.com/questions/35355930/matplotlib-figure-to-image-as-a-numpy-array
            canvas = FigureCanvasAgg(fig)

            # Retrieve a view on the renderer buffer
            canvas.draw()
            buf = canvas.buffer_rgba()

            # convert to a NumPy array
            X = np.asarray(buf)
            plt.close(fig)
            return X

        elif mode == RENDER_HUMAN:
            logger.info('step %d %s reward %f done %s' % (
                self.step_no, self.current_scan,
                self.last_reward, self.episode_done))

            fig = render_scan(self.current_scan)
            plt.show()
            plt.close(fig)

        else:
            super(DDAEnv, self).render(mode=mode)  # just raise an exception

    def write_mzML(self, out_dir, out_file):
        self.vimms_env.write_mzML(out_dir, out_file)
        self.vimms_env.write_eval_data(out_dir, out_file)
