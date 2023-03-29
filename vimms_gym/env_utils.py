import numpy as np

from vimms_gym.common import clip_value, MAX_ROI_LENGTH_SECONDS


def scale_intensities(intensity_values, num_features, max_value):
    if np.all(intensity_values == 0):
        # If all the input values are zero, return it
        return intensity_values

    # operate only on the slice actually containing peak data
    arr = intensity_values[:num_features]

    # Identify non-zero values using a boolean mask
    # Apply log transform only to non-zero values
    nonzero_mask = arr != 0

    # take the log but preserve the sign later
    log_intensity_values = np.zeros_like(arr)
    log_intensity_values[nonzero_mask] = np.log(np.abs(arr[nonzero_mask]))

    # Scale the log-transformed intensity values to be between 0 and 1
    scaled_intensity_values = log_intensity_values / max_value
    scaled_intensity_values = np.clip(scaled_intensity_values, 0, 1)

    # if the sign is initially negative, set it back
    scaled_intensity_values = np.sign(arr) * scaled_intensity_values

    # set the calculation back to intensity_values
    intensity_values[:num_features] = scaled_intensity_values
    return intensity_values


def update_feature_roi(feature, i, state):
    # for each feature, get its associated live ROI
    # there should always be a live ROI for each feature
    roi = feature.roi

    # last few intensity values of this ROI
    roi_intensities_2 = 0.0
    roi_intensities_3 = 0.0
    roi_intensities_4 = 0.0
    roi_intensities_5 = 0.0
    roi_intensities_6 = 0.0
    roi_intensities_7 = 0.0
    roi_intensities_8 = 0.0
    roi_intensities_9 = 0.0
    avg_intensity = 0.0

    # current length of this ROI (in seconds)
    try:
        roi_length = clip_value(roi.length_in_seconds, MAX_ROI_LENGTH_SECONDS)
    except AttributeError:  # no ROI object
        roi_length = 0.0

    try:
        # time elapsed (in seconds) since last fragmentation of this ROI
        val = roi.rt_list[-1] - roi.rt_list[roi.fragmented_index]
        roi_elapsed_time_since_last_frag = clip_value(np.log(val), MAX_ROI_LENGTH_SECONDS)
    except AttributeError:  # no ROI object, or never been fragmented
        roi_elapsed_time_since_last_frag = 0.0

    try:
        # intensity of this ROI at last fragmentation
        roi_intensity_at_last_frag = roi.intensity_list[roi.fragmented_index]
    except AttributeError:  # no ROI object, or never been fragmented
        roi_intensity_at_last_frag = 0.0

    try:
        # minimum intensity of this ROI since last fragmentation
        roi_min_intensity_since_last_frag = min(roi.intensity_list[roi.fragmented_index:])
    except AttributeError:  # no ROI object, or never been fragmented
        roi_min_intensity_since_last_frag = 0.0

    try:
        # maximum intensity of this ROI since last fragmentation
        roi_max_intensity_since_last_frag = max(roi.intensity_list[roi.fragmented_index:])
    except AttributeError:  # no ROI object, or never been fragmented
        roi_max_intensity_since_last_frag = 0.0

    if roi is not None:
        intensities = roi.intensity_list
        avg_intensity = np.mean(intensities)
        try:
            roi_intensities_2 = intensities[-2]
        except IndexError:
            pass

        try:
            roi_intensities_3 = intensities[-3]
        except IndexError:
            pass

        try:
            roi_intensities_4 = intensities[-4]
        except IndexError:
            pass

        try:
            roi_intensities_5 = intensities[-5]
        except IndexError:
            pass

        try:
            roi_intensities_6 = intensities[-6]
        except IndexError:
            pass

        try:
            roi_intensities_7 = intensities[-7]
        except IndexError:
            pass

        try:
            roi_intensities_8 = intensities[-8]
        except IndexError:
            pass

        try:
            roi_intensities_9 = intensities[-9]
        except IndexError:
            pass

    state['roi_length'][i] = roi_length
    state['roi_elapsed_time_since_last_frag'][i] = roi_elapsed_time_since_last_frag
    state['roi_intensity_at_last_frag'][i] = roi_intensity_at_last_frag
    state['roi_min_intensity_since_last_frag'][i] = roi_min_intensity_since_last_frag
    state['roi_max_intensity_since_last_frag'][i] = roi_max_intensity_since_last_frag

    state['roi_intensities_2'][i] = roi_intensities_2
    state['roi_intensities_3'][i] = roi_intensities_3
    state['roi_intensities_4'][i] = roi_intensities_4
    state['roi_intensities_5'][i] = roi_intensities_5
    state['roi_intensities_6'][i] = roi_intensities_6
    state['roi_intensities_7'][i] = roi_intensities_7
    state['roi_intensities_8'][i] = roi_intensities_8
    state['roi_intensities_9'][i] = roi_intensities_9

    state['avg_roi_intensities'][i] = avg_intensity


def shifted_sigmoid(x, sigmoid_range=2, sigmoid_shift=-1):
    """
    The shifted_sigmoid function is a modified sigmoid function that starts
    at 0 when x is 0 and has an upper bound of 1.0. The sigmoid function is
    transformed by scaling it with the sigmoid_range variable and shifting it
    using the sigmoid_shift variable.
    :param x: Input to the shifted_sigmoid function
    :param sigmoid_range: Scaling factor for the sigmoid function (default: 2)
    :param sigmoid_shift: Shifting factor for the sigmoid function (default: -1)
    :return: Transformed sigmoid value
    """
    return (sigmoid_range / (1 + np.exp(-x))) + sigmoid_shift
