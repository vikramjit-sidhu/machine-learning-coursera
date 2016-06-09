function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
								   
% Implements the neural network cost function for a three layer
% neural network which performs classification

	Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
					 hidden_layer_size, (input_layer_size + 1));

	Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
					 num_labels, (hidden_layer_size + 1));
					 
	Theta1_wo_bias = Theta1(:,2:end);
	Theta2_wo_bias = Theta2(:,2:end);

	m = size(X, 1);	% num training examples
	
	activation_1_all = [ones(m,1) X]';	% 401x5000
	
	z_2_all = Theta1 * activation_1_all;
	activation_2_all = sigmoid(z_2_all);	% 25x5000
	activation_2_all_wbias = [ones(1, m); activation_2_all];	% 26x5000
	
	z_3_all = Theta2 * activation_2_all_wbias;
	activation_3_all = sigmoid(z_3_all);	% 10x5000
	
	J = 0;
	Delta1 = zeros(size(Theta1));
	Delta2 = zeros(size(Theta2));
	for train_set_num = 1:m
		current_y = zeros(num_labels, 1);
		current_y(y(train_set_num)) = 1;
		
		a_3 = activation_3_all(:, train_set_num);
		
		J += ((-current_y)' * log(a_3)) - ((1-current_y)' * log(1-a_3));
		
		% Error in activations
		del_layer3 = a_3 - current_y;	% 10x1
		del_layer2 = (Theta2_wo_bias' * del_layer3) .* sigmoidGradient(z_2_all(:, train_set_num));	% 25x1
		
		Delta1 = Delta1 .+ (del_layer2 * activation_1_all(:, train_set_num)');
		Delta2 = Delta2 .+ (del_layer3 * activation_2_all_wbias(:, train_set_num)');
	end
	J = J/m;
	
	% regularize cost function
	reg_term = 0;
	reg_term += sum(diag(Theta1_wo_bias * Theta1_wo_bias'));
	reg_term += sum(diag(Theta2_wo_bias * Theta2_wo_bias'));
	J += (reg_term * lambda) / (2*m);
	
	% gradiant terms
	Theta1_grad = Delta1 / m;
	Theta2_grad = Delta2 / m;

	% regularizing gradiant terms
	Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (lambda / m) * Theta1_wo_bias;
	Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + (lambda / m) * Theta2_wo_bias;
	
	grad = [Theta1_grad(:) ; Theta2_grad(:)];	% gradiants have to be returned as a vector

end
