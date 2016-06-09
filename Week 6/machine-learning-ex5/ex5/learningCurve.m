function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
% Find the parameters of a learning function using the training set
% Plot the curve of the cost function vs the number of training examples
% Plot the above graph for the training and cross-validation sets.

	m = size(X, 1); % Number of training examples
	

	error_train = zeros(m, 1);
	error_val   = zeros(m, 1);
	
	for i = 1:m
		theta = trainLinearReg(X(1:i, :), y(1:i), lambda);
		error_train(i) = linearRegCostFunction(X(1:i, :), y(1:i), theta, 0);
		error_val(i) = linearRegCostFunction(Xval, yval, theta, 0);	% Use the entire cross-validation set for the error
	end
	
end
