% Clear the workspace and initialize consistent random values
clc;
clear;
close all;

% Load the monitor data (add name to the directory)
load("net_results\...");

% Set characteristics for plotting
line_width = 2;   
font_size = 30;

monitor_data = net_results.monitor_data;

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
xlabel('Iterations', 'FontSize', font_size);
ylabel('RMSE values', 'FontSize', font_size);
legend('Training RMSE','Training RMSE smoothed','Validation RMSE', 'Location', 'best', 'FontSize', font_size);
title('RMSE Metrics', 'FontSize', font_size);
xlim([min(monitor_data.iterations_store), max(monitor_data.iterations_store)]);
grid on;
hold off;

% Plot ISS metrics
subplot(2,1,2);

% Count the number of ISS fields in monitor_data
iss_fields = fieldnames(monitor_data);
num_iss_layers = sum(contains(iss_fields, 'iss_'));

% Plot ISS metrics dynamically based on the number of layers
colors = ['r', 'b', 'g', 'm', 'c', 'y', 'k']; % Add more colors if needed
for i = 1:num_iss_layers
    iss_field = strcat('iss_', num2str(i), '_store');
    if isfield(monitor_data, iss_field)
        plot(monitor_data.iterations_store, monitor_data.(iss_field), 'Color', colors(mod(i-1,length(colors))+1), 'DisplayName', ['ISS ', num2str(i)], 'LineWidth', line_width);
        hold on;
    end
end

xlabel('Iterations', 'FontSize', font_size);
ylabel('ISS values', 'FontSize', font_size);
legend('show', 'Location', 'best', 'FontSize', font_size);
title('ISS Metrics', 'FontSize', font_size);
xlim([min(monitor_data.iterations_store), max(monitor_data.iterations_store)]);
grid on;
hold off;

linkaxes(findall(gcf,'Type','axes'), 'x');
F.Color = 'w';

% Save the updated monitor data with interpolated values
% save(file_name,"monitor_data");