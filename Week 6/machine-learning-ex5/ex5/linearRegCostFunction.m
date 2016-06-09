function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%Finds the cost and gradient for regularized linear 
%regression with multiple variables

	m = length(y); % number of training examples

	hypothesis = X * theta;	
	pred_error = hypothesis - y;
	
	J = 1/(2*m) * (pred_error' * pred_error);
	J += lambda/(2*m) * (theta(2:end)' * theta(2:end)); 	% regularize the cost function
	
	grad = 1/m * (pred_error' * X)';
	grad(2:end) += lambda/m * theta(2:end);	% regularizing the gradiant
	grad = grad(:);

end
