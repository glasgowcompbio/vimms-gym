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


def process_array(arr):
    # Find the indices of the non-zero elements
    non_zero_indices = np.where(arr != 0)

    # Compute the log of the non-zero elements
    log_arr = np.zeros_like(arr, dtype=float)
    log_arr[non_zero_indices] = np.log(arr[non_zero_indices])

    # Find the maximum log value
    max_log_value = np.max(log_arr)

    # Divide the log values by the maximum log value
    log_arr /= max_log_value

    # Clip the results between 0 and 1
    clipped_arr = np.clip(log_arr, 0, 1)

    return clipped_arr


def normalize_roi_data(roi_data):
    # find the maximum value for each ROI along the timepoint axis
    max_vals = np.max(roi_data, axis=1)

    # skip any ROIs that are all zeros
    zero_mask = max_vals != 0
    roi_data = roi_data[zero_mask]
    max_vals = max_vals[zero_mask]

    # divide each ROI by its respective maximum value
    normalized_data = roi_data / max_vals[:, np.newaxis]

    # flip the data, so last column is the most recent ROI point
    normalized_data = np.flip(normalized_data)

    return normalized_data


def update_feature_roi(feature, i, state):
    roi = feature.roi

    roi_length = 0.0

    if roi is not None:
        roi_length = clip_value(roi.length_in_seconds, MAX_ROI_LENGTH_SECONDS)

        if hasattr(roi, 'rt_list') and hasattr(roi, 'fragmented_index'):
            val = roi.rt_list[-1] - roi.rt_list[roi.fragmented_index]

        intensities = roi.intensity_list

        for j in range(1, 11):
            try:
                state['_roi_intensities'][i][j - 1] = intensities[-j]
            except IndexError:
                state['_roi_intensities'][i][j - 1] = 0.0

    state['roi_length'][i] = roi_length


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


class RoiTracker:
    def __init__(self, max_rois, num_features):
        self.max_rois = max_rois
        self.num_features = num_features
        self.rois = []

    def update_roi(self, roi):
        # Update the list of ROIs with new information
        index = self.get_roi_index(roi.id)
        if index is None:
            self.rois.append(roi)
        else:
            self.rois[index] = roi

    def get_roi_index(self, roi_id):
        for index, roi in enumerate(self.rois):
            if roi.id == roi_id:
                return index
        return None

    def select_rois(self):
        # Implement the selection mechanism based on the criterion of your choice
        # Example: Select top-N ROIs based on intensity
        sorted_rois = sorted(self.rois, key=lambda roi: roi.intensity_list[-1], reverse=True)
        return sorted_rois[:self.max_rois]

    def get_state_matrix(self):
        selected_rois = self.select_rois()
        state_matrix = np.zeros((self.max_rois, self.num_features + 1))

        for i, roi in enumerate(selected_rois):
            roi_features = [roi['id'], roi['intensity'], roi['mz'], roi['rt'],
                            roi['fragmentation_status']]
            state_matrix[i, :-1] = roi_features
            state_matrix[i, -1] = 1  # Set the presence indicator to 1

        return state_matrix


def extract_value(variable):
    if isinstance(variable, np.int64):
        return variable
    elif isinstance(variable, int):
        return variable
    elif isinstance(variable, np.ndarray) and len(variable) == 1:
        return variable[0]
    raise ValueError('Unexpected variable type')