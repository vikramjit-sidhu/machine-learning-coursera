function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

	m = size(X, 1);	% num training examples
	num_labels = size(Theta2, 1);

	% the first activation function is the input parameters
	input_second = [ones(m,1) X]';
	activation_second = sigmoid(Theta1 * input_second);
	
	input_third = [ones(1,m); activation_second];
	activation_third = (sigmoid(Theta2 * input_third))';	% m x num_labels

	[max_hypothesis, p] = max(activation_third, [], 2);

end
