function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
	
	partial_derivate_theta1 = 0;
	partial_derivate_theta2 = 0;
	for i = 1:m
		hypothesis = theta' * X(i,:)';
		partial_derivate_theta1 += hypothesis - y(i,:);
		partial_derivate_theta2 += (hypothesis - y(i,:)) * X(i,2);
	end	
	
	theta1_new = theta(1) - ((alpha * partial_derivate_theta1) / m);
	theta2_new = theta(2) - ((alpha * partial_derivate_theta2) / m);
	theta = [theta1_new; theta2_new];

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
