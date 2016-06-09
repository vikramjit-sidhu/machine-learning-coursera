function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

sq_diff_sum = 0;	%sum of differences in the hypothesis function and actual data point

for i = 1:m
	hypothesis = theta' * X(i,:)';
	sq_diff_sum += (hypothesis - y(i,:)) ^ 2;
end;

J = sq_diff_sum / (2*m);

% =========================================================================

end
