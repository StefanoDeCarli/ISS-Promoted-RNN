% Custom loss function for iss-promoted GRU networks
function [loss, gradients, state, iss, mse_loss] = clf_gru(net, x, targets, num_layers, epsilon, penalty, u_max)

    % Forward pass through the network
    [Y, state] = forward(net, x);
    
    % Initialize arrays for ISS and ISS penalties
    iss_cell = cell(num_layers,1);       % Initialize all iss condition values
    iss_penalties = cell(num_layers,1);  % Initialize all iss penalty values

    % Loop through each GRU layer to extract weights and compute ISS
    for i = 1:num_layers
        idx = 3 * (i - 1) + 1;
        dummy_weight = net.Learnables.Value(idx);
        num_units = size(dummy_weight{1}, 1) / 3; % Assuming 3 gates, determine the number of units in the layer

        % Extract weights from net
        [W_r, ~, ~] = split_gru_weights(net.Learnables.Value(idx), num_units);          % [W_r, W_z, W_h] - Concatenated input weights
        [R_r, ~, R_h] = split_gru_weights(net.Learnables.Value(idx + 1), num_units);    % [R_r, R_z, R_h] - Concatenated recurrent weights
        [b_r, ~, ~] = split_gru_biases(net.Learnables.Value(idx + 2), num_units);       % [b_r, b_z, b_h] - Concatenated biases for GRU

        % Initialize parameters for stability
        r_parameters = [W_r*u_max{i} R_r b_r];

        sig_r = sigmoid(vecnorm(vecnorm(r_parameters, inf, 1), inf, 2));

        % Compute the ISS for the current layer
        iss_cell{i} = sig_r * vecnorm(vecnorm(R_h, inf, 1), inf, 2);

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

% Function to split gru weights
function [W_r, W_z, W_h] = split_gru_weights(W, num_units)
    W_r = W{1}(1:num_units, :);
    W_z = W{1}(num_units+1:2*num_units, :);
    W_h = W{1}(2*num_units+1:end, :);
end

% Function to split gru biases
function [b_r, b_z, b_h] = split_gru_biases(b, num_units)
    b_r = b{1}(1:num_units);
    b_z = b{1}(num_units+1:2*num_units);
    b_h = b{1}(2*num_units+1:end);
end