% Function to create the monitor dependant on the number of layers
function monitor = generate_monitor(num_layers)
    % Set iss metrics based on the number of layers
    iss_metrics = arrayfun(@(i) "iss_" + i, 1:num_layers, 'UniformOutput', false);

    % Create the training progress monitor with the appropriate configuration
    monitor = trainingProgressMonitor( ...
        Metrics=["TrainingRMSE", "ValidationRMSE", "TrainingRMSE_smooth", iss_metrics], ...
        Info=["LearnRate", "Epoch", "Iteration", "ExecutionEnvironment"], ...
        XLabel="Iteration");

    % Configure subplot groups for RMSE and ISS
    groupSubPlot(monitor, "RMSE", ["TrainingRMSE", "TrainingRMSE_smooth", "ValidationRMSE"]);
    groupSubPlot(monitor, "ISS", iss_metrics);
end
