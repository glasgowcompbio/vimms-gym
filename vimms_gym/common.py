import numpy as np

MS1_REWARD = 0.1
REPEATED_MS1_REWARD = -0.1
REPEATED_FRAG_REWARD = -0.2
INVALID_MOVE_REWARD = -1.0

MAX_REPEATED_FRAGS_ALLOWED = 5
MAX_OBSERVED_LOG_INTENSITY = np.log(1E20)
MAX_ROI_LENGTH_SECONDS = 100


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
