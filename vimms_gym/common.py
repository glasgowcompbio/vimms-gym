import numpy as np
import pylab as plt

MS1_REWARD = 0.1
INVALID_MOVE_REWARD = -1.0
MAX_OBSERVED_LOG_INTENSITY = np.log(1E20)
MAX_ROI_LENGTH_SECONDS = 100

ALPHA = 0.5

METHOD_RANDOM = 'random'
METHOD_FULLSCAN = 'fullscan'
METHOD_TOPN = 'topN'
METHOD_PPO = 'PPO'
METHOD_DQN = 'DQN'

RENDER_HUMAN = 'human'
RENDER_RGB_ARRAY = 'rgb_array'


def clip_value(value, max_value, min_range=0.0, max_range=1.0):
    '''
    Scale value by max_value, then clip it to [-min_range, max_range]
    '''
    value = min(value, max_value) if value >= 0 else max(value, -max_value)
    value = value / max_value
    if value < min_range:
        value = min_range
    elif value > max_range:
        value = max_range
    return value


def render_scan(scan):
    if scan is None:
        return None

    fig = plt.figure()
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
    return fig


def linear_schedule(initial_value, min_value=0.0):
    def func(progress_remaining):
        return max(progress_remaining * initial_value, min_value)

    return func
