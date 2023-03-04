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
    log_intensity_values = np.zeros_like(arr)
    log_intensity_values[nonzero_mask] = np.log(arr[nonzero_mask])

    # Scale the log-transformed intensity values to be between 0 and 1
    scaled_intensity_values = log_intensity_values / max_value
    scaled_intensity_values = np.clip(scaled_intensity_values, 0, 1)

    # set the calculation back to intensity_values
    intensity_values[:num_features] = scaled_intensity_values
    return intensity_values


def update_feature_roi(feature, i, state):
    # for each feature, get its associated live ROI
    # there should always be a live ROI for each feature
    roi = feature.roi

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

    # last few intensity values of this ROI
    roi_intensities_2 = 0.0
    roi_intensities_3 = 0.0
    roi_intensities_4 = 0.0
    roi_intensities_5 = 0.0

    if roi is not None:
        intensities = roi.intensity_list
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

    state['roi_length'][i] = roi_length
    state['roi_elapsed_time_since_last_frag'][i] = roi_elapsed_time_since_last_frag
    state['roi_intensity_at_last_frag'][i] = roi_intensity_at_last_frag
    state['roi_min_intensity_since_last_frag'][i] = roi_min_intensity_since_last_frag
    state['roi_max_intensity_since_last_frag'][i] = roi_max_intensity_since_last_frag

    state['roi_intensities_2'][i] = roi_intensities_2
    state['roi_intensities_3'][i] = roi_intensities_3
    state['roi_intensities_4'][i] = roi_intensities_4
    state['roi_intensities_5'][i] = roi_intensities_5
