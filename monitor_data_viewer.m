% Clear the workspace and initialize consistent random values
clc;
clear;
close all;

% Load the monitor data (add name to the directory)
load("net_results\...");

% Set characteristics for plotting
line_width = 2;   
font_size = 12;
set(groot, 'defaultAxesTickLabelInterpreter','latex');
set(groot, 'defaultLegendInterpreter','latex');

monitor_data = net_results.monitor_data;

% Find the first occurrence of zero in rmse_train
zero_idx = find(monitor_data.rmse_train == 0, 1);

% If there is a zero in rmse_train, cut the data
if ~isempty(zero_idx)
    monitor_data.rmse_train = monitor_data.rmse_train(1:zero_idx-1);
    monitor_data.rmse_train_smooth = monitor_data.rmse_train_smooth(1:zero_idx-1);
    monitor_data.rmse_validation = monitor_data.rmse_validation(1:zero_idx-1);
    monitor_data.iterations_store = monitor_data.iterations_store(1:zero_idx-1);
    monitor_data.iss_store = monitor_data.iss_store(1:zero_idx-1,:);
end

% Interpolate RMSE smooth and validation values
valid_smooth_idx = monitor_data.rmse_train_smooth > 0;
valid_validation_idx = monitor_data.rmse_validation > 0;

interp_rmse_train_smooth = interp1(monitor_data.iterations_store(valid_smooth_idx), monitor_data.rmse_train_smooth(valid_smooth_idx), monitor_data.iterations_store, 'linear', 'extrap');
interp_rmse_validation = interp1(monitor_data.iterations_store(valid_validation_idx), monitor_data.rmse_validation(valid_validation_idx), monitor_data.iterations_store, 'linear', 'extrap');

% Save the interpolated data back to monitor_data
monitor_data.rmse_train_smooth = interp_rmse_train_smooth;
monitor_data.rmse_validation = interp_rmse_validation;

% Plot RMSE metrics
F = figure;
subplot(2,1,1);
plot(monitor_data.iterations_store, monitor_data.rmse_train, 'c', 'DisplayName', 'Training RMSE', 'LineWidth', line_width);
hold on;
plot(monitor_data.iterations_store, interp_rmse_train_smooth, 'b', 'DisplayName', 'Training RMSE Smooth', 'LineWidth', line_width);
plot(monitor_data.iterations_store, interp_rmse_validation, 'r', 'DisplayName', 'Validation RMSE', 'LineWidth', line_width);
xlabel('Iterations', 'FontSize', font_size, 'Interpreter', 'latex');
ylabel('RMSE values', 'FontSize', font_size, 'Interpreter', 'latex');
legend('Training RMSE','Training RMSE smoothed','Validation RMSE', 'Location', 'best', 'FontSize', font_size, 'Interpreter', 'latex');
title('RMSE Metrics', 'FontSize', font_size, 'Interpreter', 'latex');
xlim([min(monitor_data.iterations_store), max(monitor_data.iterations_store)]);
grid on;

% Plot a vertical dotted line at the lowest validation score iteration
xline(monitor_data.min_val_iteration, '--k', 'LineWidth', line_width, 'DisplayName',"Min ISS evaluation rmse"); % Assuming all seeds are equal

hold off;

% Plot ISS metrics
subplot(2,1,2);

% Count the number of ISS fields in monitor_data
num_iss_layers = width(monitor_data.iss_store);

% Plot ISS metrics dynamically based on the number of layers
colors = ['r', 'b', 'g', 'm', 'c', 'y', 'k']; % Add more colors if needed
for i = 1:num_iss_layers
    iss_field = strcat('iss_', num2str(i), '_store');
    plot(monitor_data.iterations_store, monitor_data.iss_store(:,i), 'Color', colors(mod(i-1,length(colors))+1), 'DisplayName', ['ISS ', num2str(i)], 'LineWidth', line_width);
    hold on;
end

xlabel('Iterations', 'FontSize', font_size, 'Interpreter', 'latex');
ylabel('ISS values', 'FontSize', font_size, 'Interpreter', 'latex');
legend('show', 'Location', 'best', 'FontSize', font_size, 'Interpreter', 'latex');
title('ISS Metrics', 'FontSize', font_size, 'Interpreter', 'latex');
xlim([min(monitor_data.iterations_store), max(monitor_data.iterations_store)]);
grid on;

% Plot a vertical dotted line at the lowest validation score iteration
xline(monitor_data.min_val_iteration, '--k', 'LineWidth', line_width, 'DisplayName',"Min ISS evaluation rmse", 'Interpreter', 'latex'); % Assuming all seeds are equal
hold off;

linkaxes(findall(gcf,'Type','axes'), 'x');
F.Color = 'w';