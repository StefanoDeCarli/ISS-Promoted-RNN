% Function to convert batches of training data into dlarray format
function [dlx, dly] = preprocess_mini_batches(x_train_batch, y_train_batch)
    num_batches = size(x_train_batch,1);  % Total number of batches
    dlx = cell(num_batches, 1);  % Preallocate cell array for input dlarrays
    dly = cell(num_batches, 1);  % Preallocate cell array for output dlarrays

    for i = 1:num_batches
        % Extract the i-th batch of inputs and outputs
        x_batch = x_train_batch{i};
        y_batch = y_train_batch{i};

        num_sequences = size(x_batch,1);            % Number of sequences per batch
        num_time_steps = zeros(numel(x_batch),1);   % Initialize number of time steps per sequence
        num_features = size(x_batch{1},2);          % Number of features per time step
        
        max_time_step = 0;

        for j = 1:numel(x_batch)
            num_time_steps(j) = size(x_batch{j},1);
            if num_time_steps(j) > max_time_step
                max_time_step = num_time_steps(j);
            end
        end

        % Preallocate arrays to hold reshaped data for both x and y
        x_data = zeros(max_time_step, num_sequences, num_features);
        y_data = zeros(max_time_step, num_sequences, size(y_batch{1}, 2));

        for j = 1:num_sequences
            x_data(:,j,:) = paddata(x_batch{j},max_time_step);  % Reshape each sequence into preallocated arrays
            y_data(:,j,:) = paddata(y_batch{j},max_time_step);
        end

        % Convert reshaped data into dlarray objects with the
        % 'TBC' (Time, Batch, Channel (features)) format
        dlx{i} = dlarray(x_data, 'TBC');
        dly{i} = dlarray(y_data, 'TBC');
    end
end