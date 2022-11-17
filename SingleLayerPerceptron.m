%/*******************************************************
% * Copyright (C) 2019-2020 Ruixu Liu <liur05@udayton.edu>
% * 
% * This file is part of Artificial Neural Network.
% * 
% * MIT License
% *******************************************************/
%% Single Layer Perceptron
function [Error_for_plot, w_ji,Total_iterations] = SingleLayerPerceptron(input_data, labels, input_neurons,...
    output_neurons,learning_rate, Error_threshold, sigmoid_fct)
    %Input:  input_data            784 x number of training data
    %        labels                number of training data  x 1
    %        input_neurons         784
    %        output_neurons        10
    %        learning_rate         0.5, 0.1, 0.01
    %        Error_threshold       0.001
    
    %Output: Error_for_plot        MSE for each iteration
    %        w_ji                  The weights of this network
    
    w_ji = ones(input_neurons,output_neurons)/2;
    iterations = 1;
    Error_end = 1;
    max_iterations = 1000;
    Error_update = zeros(max_iterations,1);

    while Error_end > Error_threshold && iterations < max_iterations
        E = 0;
        for m = 1 : size(input_data,2)
            % ex: from m=s to P
            
            d = zeros(1,output_neurons);
            y = zeros(1,output_neurons);
            x_i = repmat(input_data(:,m),[1,output_neurons]);
            
            Net = sum(w_ji .* x_i);
            
            if sigmoid_fct == "on"
                y = (1)./(1+exp(-Net));
            else
            y(Net > 0) = 1; % 0 is the threshold
            end

            d(labels(m)+1) = 1;

            sigma = d-y;
            sigma_ji = repmat(sigma,[input_neurons,1]);
            
            delta_w = learning_rate*sigma_ji.*x_i;

            w_ji = w_ji + delta_w;

            En = sum((d-y).^2);
            E = E + En;
        end

        Error_end = E/(10*size(input_data,2)); % MSE: accounting for M and P
        Error_update(iterations) = Error_end;
        iterations = iterations+1;
    end
    
    Error_for_plot = Error_update(1:iterations-1);
    Total_iterations = iterations-1;
end
