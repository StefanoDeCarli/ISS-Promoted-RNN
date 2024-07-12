% Function to create mini batches in training and validation
function [mini_batch_input, mini_batch_output] = create_mini_batches(inputs, outputs, mini_batch_size)
    num_observations = length(inputs);
    idx = randperm(num_observations);
    
    mini_batch_input = cell(floor(num_observations / mini_batch_size),1);
    mini_batch_output = cell(floor(num_observations / mini_batch_size),1);
    
    for i = 1:floor(num_observations / mini_batch_size)
        start_idx = (i-1) * mini_batch_size + 1;
        end_idx = i * mini_batch_size;
        
        batch_idx = idx(start_idx:end_idx);
        mini_batch_input{i} = inputs(batch_idx);
        mini_batch_output{i} = outputs(batch_idx);
    end
end