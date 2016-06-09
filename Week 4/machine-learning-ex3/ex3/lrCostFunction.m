function [J, grad] = lrCostFunction(theta, X, y, lambda)
% A vectorized version of logistic regression with regularization

	m = length(y); % number of training examples
	n = size(X,2);	% number of features

	hypothesis = sigmoid(X * theta);
	
	cost = ((-y)' * log(hypothesis)) + ((y-1)' * log(1-hypothesis));
	J = cost / m;
	reg_cost = (lambda / (2 * m)) * (theta(2:n)' * theta(2:n));
	J += reg_cost; 	% regularizing cost function
	
	diff = hypothesis - y;
	grad = ((diff' * X)') / m;
	grad(2:n) += (lambda / m) * theta(2:n);	% regularizing gradiant

end
