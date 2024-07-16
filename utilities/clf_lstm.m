% Custom loss function for iss-promoted LSTM networks
function [loss, gradients, state, iss, mse_loss] = clf_lstm(net, x, targets, num_layers, epsilon, penalty, u_max)

    % Forward pass through the network
    [Y, state] = forward(net, x);
    
    % Initialize arrays for ISS and ISS penalties
    iss_cell = cell(num_layers,1);          % Initialize all iss condition values
    iss_penalties = cell(num_layers,1);     % Initialize all iss penalty values

    % Loop through each LSTM layer to extract weights and compute ISS
    for i = 1:num_layers
        idx = 3 * (i - 1) + 1;
        dummy_weight = net.Learnables.Value(idx);
        num_units = size(dummy_weight{1}, 1) / 4; % Assuming 4 gates, determine the number of units in the layer

        % Extract weights from net
        [W_i, W_f, ~, ~] = split_lstm_weights(net.Learnables.Value(idx), num_units);        % [W_i W_f W_g W_o] - Concatenated input weights
        [R_i, R_f, R_g, ~] = split_lstm_weights(net.Learnables.Value(idx + 1), num_units);  % [R_i R_f R_g R_o] - Concatenated recurrent weights
        [b_i, b_f, ~, ~] = split_lstm_biases(net.Learnables.Value(idx + 2), num_units);     % [b_i b_f b_g b_o] - Concatenated biases for LSTM

        % Initialize parameters for stability
        f_parameters = [W_f*u_max{i} R_f b_f];
        i_parameters = [W_i*u_max{i} R_i b_i];

        sig_f = sigmoid(vecnorm(vecnorm(f_parameters, inf, 1), inf, 2));
        sig_i = sigmoid(vecnorm(vecnorm(i_parameters, inf, 1), inf, 2));

        % Compute the ISS for the current layer
        iss_cell{i} = sig_f + sig_i * vecnorm(vecnorm(R_g, inf, 1), inf, 2);

        % Compute ISS penalty for the current layer
        iss_penalties{i} = penalty * max(0, iss_cell{i} - 1 + epsilon);
    end

    % Calculate squared differences
    squared_diff = (Y - targets).^2;

    % Compute mean over all elements
    mse_loss = sum(squared_diff, 'all') / numel(Y);

    % Initialize total loss
    loss = mse_loss;

    % Total loss is the sum of MSE loss and ISS penalties
    for i = 1:num_layers
        loss = loss + iss_penalties{i};
    end

    % Check for GPU availability and convert to gpuArray if possible
    if canUseGPU() % Using a helper function to check GPU availability
        loss = gpuArray(loss);
    end

    % Compute gradients of the loss with respect to the learnable parameters
    gradients = dlgradient(loss, net.Learnables);

    % Convert extracted values to double if on GPU
    iss = zeros(num_layers,1);
    for i = 1:num_layers
        iss(i) = double(iss_cell{i});
    end

    mse_loss = double(extractdata(mse_loss));
end

% Function to split lstm weights
function [W_i, W_f, W_g, W_o] = split_lstm_weights(W, num_units)
    W_i = W{1}(1:num_units, :);
    W_f = W{1}(num_units+1:2*num_units, :);
    W_g = W{1}(2*num_units+1:3*num_units, :);
    W_o = W{1}(3*num_units+1:end, :);
end

% Function to split lstm biases
function [b_i, b_f, b_g, b_o] = split_lstm_biases(b, num_units)
    b_i = b{1}(1:num_units);
    b_f = b{1}(num_units+1:2*num_units);
    b_g = b{1}(2*num_units+1:3*num_units);
    b_o = b{1}(3*num_units+1:end);
end