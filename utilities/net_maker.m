% Function to create the layers of the network based on specifications
function network_layers = net_maker(is_lstm, hidden_units, num_features, num_responses, dropout_rate)
    % Determine the number of layers from the length of the hidden_units array
    num_layers = height(hidden_units);
    
    % Initialize the network layers array
    network_layers = [sequenceInputLayer(num_features, 'Name', 'input')];

    % Add LSTM or GRU layers based on the is_lstm flag
    for i = 1:num_layers
        if is_lstm
            layer_name = ['lstm_', num2str(i)];
            network_layers = [network_layers, ...
                lstmLayer(hidden_units(i), 'OutputMode', 'sequence', 'Name', layer_name), ...
                dropoutLayer(dropout_rate, 'Name', ['drop_', num2str(i)])];
        else
            layer_name = ['gru_', num2str(i)];
            network_layers = [network_layers, ...
                gruLayer(hidden_units(i), 'OutputMode', 'sequence', 'Name', layer_name), ...
                dropoutLayer(dropout_rate, 'Name', ['drop_', num2str(i)])];
        end
    end

    % Add the final fully connected layer
    network_layers = [network_layers, ...
        fullyConnectedLayer(num_responses, 'Name', 'fc')];
end