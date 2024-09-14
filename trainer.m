% Clear the workspace and initialize consistent random values
clc;
clear;
close all;
random_seed = 1;
rng(random_seed);

% Load dataset
data = load(fullfile('data', 'SMI_data.mat'));
varName = fieldnames(data);   % Get the field name(s) in the structure
data = data.(varName{1});     % Access the contents using dynamic field referencing

train_dataset = data.train_30s;
valid_dataset = data.validation_30s;

train_dataset.x = transpose_cell(train_dataset.x);
train_dataset.y = transpose_cell(train_dataset.y);

valid_dataset.x = transpose_cell(valid_dataset.x);
valid_dataset.y = transpose_cell(valid_dataset.y);

is_lstm = true;

hidden_units = [256;128;128]; % X layers
dropout_rate = 0.4;

u_max_inputs = [2;2;2;2;2;2;3];
learn_rate = 0.002;
max_epochs = 5000;
mini_batch = numel(train_dataset.x); % Take all the trials, to change in case

[net,info,monitor,net_name] = ISS_train(train_dataset, valid_dataset, ... % Data
    is_lstm, hidden_units, dropout_rate, ... % Architecture
    u_max_inputs, learn_rate, max_epochs, mini_batch); % Training;

%% NET SAVE
% Initialize the net_results struct
net_results = struct(...
    'net', net, ...
    'info', info, ...
    'monitor_data', monitor);

save(['net_results/', net_name], 'net_results');

%% Useful functions for the test
function cell_array = transpose_cell(cell_array)
    for i = 1:length(cell_array)
        cell_array{i} = transpose(cell_array{i});
    end 
end