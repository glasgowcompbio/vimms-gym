import torch
import torch.nn as nn

from vimms_gym.experiments import ENV_QCB_MEDIUM_EXTRACTED, ENV_QCB_LARGE_EXTRACTED, \
    ENV_QCB_SMALL_EXTRACTED, ENV_QCB_SMALL_GAUSSIAN, ENV_QCB_MEDIUM_GAUSSIAN, \
    ENV_QCB_LARGE_GAUSSIAN

QNETWORK_CNN = 'CNN'
QNETWORK_LSTM = 'LSTM'
QNETWORK_DENSE = 'DENSE'
QNETWORK_DENSE_FLAT = 'DENSE_FLAT'


def get_QNetwork(qnetwork_type, envs, device, task, initialise=True):
    assert qnetwork_type in [QNETWORK_CNN, QNETWORK_LSTM, QNETWORK_DENSE, QNETWORK_DENSE_FLAT]
    Model = {
        QNETWORK_CNN: QNetworkCNN,
        QNETWORK_LSTM: QNetworkLSTM,
        QNETWORK_DENSE: QNetworkDense,
        QNETWORK_DENSE_FLAT: QNetworkDenseFlat
    }[qnetwork_type]

    if initialise:
        n_total_features = get_n_total_features(task)
        return Model(envs, n_total_features).to(device)
    else:
        return Model


def get_n_total_features(task):
    # FIXME: not the best way to do this ...
    n_roi_features = 300
    if task in [ENV_QCB_SMALL_GAUSSIAN, ENV_QCB_SMALL_EXTRACTED]:
        n_total_features = n_roi_features + 157
    elif task in [ENV_QCB_MEDIUM_GAUSSIAN, ENV_QCB_MEDIUM_EXTRACTED]:
        n_total_features = n_roi_features + 157
    elif task in [ENV_QCB_LARGE_GAUSSIAN, ENV_QCB_LARGE_EXTRACTED]:
        n_total_features = n_roi_features + 1207
    else:
        raise ValueError('Unknown task')
    return n_total_features


class QNetworkCNN(nn.Module):
    def __init__(self, env, n_total_features):
        super().__init__()

        self.n_hidden = [256, 256]
        self.roi_network_out = 64
        self.n_total_features = n_total_features
        self.n_roi = 30
        self.roi_length = 10
        self.n_roi_features = self.n_roi * self.roi_length  # 30 rois, each is length 10, so total is 300 features
        self.n_other_features = self.n_total_features - self.n_roi_features  # the remaining, which is 247 features

        # configuration 1

        self.roi_network = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten(start_dim=1),
            nn.Linear(64, self.roi_network_out),
            nn.ReLU(),
        )

        input_size = (self.roi_network_out * self.n_roi) + self.n_hidden[1]
        output_size = env.single_action_space.n
        self.output_layer = nn.Linear(input_size, output_size)

        self.other_network = nn.Sequential(
            nn.Linear(self.n_other_features, self.n_hidden[0]),
            nn.ReLU(),
            nn.Linear(self.n_hidden[0], self.n_hidden[0]),
            nn.ReLU(),
            nn.Linear(self.n_hidden[0], self.n_hidden[1]),
            nn.ReLU(),
        )

    def forward(self, x):
        # get dense network prediction for other features
        other_inputs = x[:, self.n_roi_features:]
        other_output = self.other_network(other_inputs)

        # transform ROI input to the right shape: (self.n_roi, self.roi_length)
        roi_inputs = x[:, 0:self.n_roi_features]
        roi_img_inputs = roi_inputs.view(-1, self.n_roi, self.roi_length)

        # Reshape the tensor to (batch_size * num_roi, 1, num_features)
        roi_img_inputs_reshaped = roi_img_inputs.reshape(-1, 1, self.roi_length)

        # Process each ROI separately
        roi_output = self.roi_network(roi_img_inputs_reshaped)

        # Reshape the output
        # flatten the output for all ROIs
        roi_output = roi_output.view(other_output.shape[0], -1)

        # average across ROIs -- doesnt' work well
        # roi_output = roi_output.view(-1, self.n_roi, self.n_hidden[1])
        # roi_output = torch.mean(roi_output, dim=1)

        # Concatenate the outputs of the two networks
        combined_output = torch.cat((roi_output, other_output), dim=-1)

        # Generate Q-value predictions
        q_values = self.output_layer(combined_output)
        return q_values


class QNetworkDense(nn.Module):
    def __init__(self, env, n_total_features):
        super().__init__()

        self.n_hidden = [256, 256]
        self.roi_network_out = 64
        self.n_total_features = n_total_features
        self.n_roi = 30
        self.roi_length = 10
        self.n_roi_features = self.n_roi * self.roi_length  # 30 rois, each is length 10, so total is 300 features
        self.n_other_features = self.n_total_features - self.n_roi_features  # the remaining, which is 247 features

        # configuration 1

        self.roi_network = nn.Sequential(
            nn.Linear(self.n_roi_features, self.n_hidden[0]),
            nn.ReLU(),
            nn.Linear(self.n_hidden[0], self.n_hidden[0]),
            nn.ReLU(),
            nn.Linear(self.n_hidden[0], self.n_hidden[1]),
            nn.ReLU(),
        )

        input_size = self.n_hidden[1] + self.n_hidden[1]
        output_size = env.single_action_space.n
        self.output_layer = nn.Linear(input_size, output_size)

        self.other_network = nn.Sequential(
            nn.Linear(self.n_other_features, self.n_hidden[0]),
            nn.ReLU(),
            nn.Linear(self.n_hidden[0], self.n_hidden[0]),
            nn.ReLU(),
            nn.Linear(self.n_hidden[0], self.n_hidden[1]),
            nn.ReLU(),
        )

    def forward(self, x):
        # get dense network prediction for other features
        other_inputs = x[:, self.n_roi_features:]
        other_output = self.other_network(other_inputs)

        # get dense network prediction for ROI features
        roi_inputs = x[:, 0:self.n_roi_features]
        roi_output = self.roi_network(roi_inputs)

        # Concatenate the outputs of the two networks
        combined_output = torch.cat((roi_output, other_output), dim=-1)

        # Generate Q-value predictions
        q_values = self.output_layer(combined_output)
        return q_values


class QNetworkDenseFlat(nn.Module):
    def __init__(self, env, n_total_features):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_total_features, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x)


class QNetworkLSTM(nn.Module):
    def __init__(self, env, n_total_features, n_hidden=[256, 256], lstm_size=64, lstm_fc_size=32,
                 n_roi=30, roi_length=10, aggregation_method=None):
        super().__init__()

        self.n_hidden = n_hidden
        self.lstm_size = lstm_size
        self.lstm_fc_size = lstm_fc_size
        self.n_total_features = n_total_features
        self.n_roi = n_roi
        self.roi_length = roi_length
        self.n_roi_features = self.n_roi * self.roi_length
        self.n_other_features = self.n_total_features - self.n_roi_features
        self.aggregation_method = aggregation_method

        # LSTM for processing ROIs
        self.roi_lstm = nn.LSTM(input_size=1, hidden_size=self.lstm_size,
                                num_layers=1, batch_first=True)
        self.roi_fc = nn.Sequential(
            nn.Linear(lstm_size, lstm_fc_size),
            nn.ReLU(),
        )

        # Dense network for processing other features
        self.other_network = nn.Sequential(
            nn.Linear(self.n_other_features, n_hidden[0]),
            nn.ReLU(),
            nn.Linear(n_hidden[0], n_hidden[0]),
            nn.ReLU(),
            nn.Linear(n_hidden[0], n_hidden[1]),
            nn.ReLU(),
        )

        mul = n_roi if aggregation_method is None else 1
        input_size = (lstm_fc_size * mul) + n_hidden[1]
        output_size = env.single_action_space.n
        self.output_layer = nn.Linear(input_size, output_size)

    def forward(self, x):
        # get dense network prediction for other features
        other_inputs = x[:, self.n_roi_features:]
        other_output = self.other_network(other_inputs)

        # reshape ROI input for LSTM: (batch_size * self.n_roi, self.roi_length, 1)
        roi_inputs = x[:, 0:self.n_roi_features]
        batch_size = x.shape[0]
        embedding_size = 1
        roi_img_inputs = roi_inputs.reshape(batch_size * self.n_roi, self.roi_length,
                                            embedding_size)

        # Compute the sequence lengths from roi_img_inputs
        seq_lengths = (roi_img_inputs != 0).sum(dim=-2)
        seq_lengths = seq_lengths.view(-1)  # flatten it
        seq_lengths = seq_lengths.cpu()

        # Prepare input for the LSTM
        lstm_input = nn.utils.rnn.pack_padded_sequence(
            roi_img_inputs, seq_lengths, batch_first=True, enforce_sorted=False)

        # Initialize hidden states for LSTM
        num_layers = self.roi_lstm.num_layers
        num_directions = 2 if self.roi_lstm.bidirectional else 1
        first_dim = num_layers * num_directions
        device = x.device
        h_t = torch.zeros((first_dim, batch_size * self.n_roi, self.lstm_size), device=device)
        h_c = torch.zeros((first_dim, batch_size * self.n_roi, self.lstm_size), device=device)

        # Process ROIs with LSTM
        lstm_outs, (h_t, h_c) = self.roi_lstm(lstm_input, (h_t.detach(), h_c.detach()))

        # Not necessary since we don't use lstm_outs, only h_t
        # lstm_outs, _ = nn.utils.rnn.pad_packed_sequence(lstm_outs)

        # Pass LSTM output through a fully connected layer
        h_t_reshaped = h_t.reshape(-1, h_t.shape[2])
        roi_output = self.roi_fc(h_t_reshaped)

        # Reshape the ROI output to have the same number of samples as the input
        roi_output_reshaped = roi_output.reshape(batch_size, self.n_roi, self.lstm_fc_size)

        # Aggregate ROI information across the ROIs dimension. For each sample in the batch,
        # summarise the information from all the ROIs into a single representation that can be
        # combined with other_output. We can use the mean to capture the average behavior
        # across all ROIs, or use the max can help the model focus on the most important ROI.
        # Aggregate the information from all the ROIs
        if self.aggregation_method == 'mean':
            roi_output_aggregated = roi_output_reshaped.mean(dim=1)
        elif self.aggregation_method == 'max':
            roi_output_aggregated = roi_output_reshaped.max(dim=1)[0]
        elif self.aggregation_method is None:  # flatten the output for all ROIs
            roi_output_aggregated = roi_output.view(batch_size, -1)

        # Concatenate the outputs of the two networks
        combined_output = torch.cat((roi_output_aggregated, other_output), dim=-1)

        # Generate Q-value predictions
        q_values = self.output_layer(combined_output)
        return q_values
