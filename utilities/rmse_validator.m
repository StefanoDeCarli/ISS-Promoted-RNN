% Function to compute the validation RMSE by receiving the whole batches
function validation_rmse = rmse_validator(net,x_val,y_val)
    
    num_batches = size(x_val,1);
    sum_rmse = 0;

    for i = 1:num_batches
        % Reset network state
        net = resetState(net);
        % Predict the system response using the network
        predicted_response = predict(net, x_val{i});
        % Compute new rmse
        rmse = sqrt(mean((y_val{i} - predicted_response).^2));
        sum_rmse = sum_rmse + rmse;
    end

    validation_rmse = mean(sum_rmse / num_batches);
end