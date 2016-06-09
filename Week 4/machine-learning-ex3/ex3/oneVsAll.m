function [all_theta] = oneVsAll(X, y, num_labels, lambda)

% Build num_labels classifiers, each to predict a value in it
% The classifiers are represented by their parameters, 
% which is returned in all_theta


	m = size(X, 1);	% num training examples
	n = size(X, 2);	% num features

	% Add ones to the X data matrix
	X = [ones(m, 1) X];

	all_theta = zeros(num_labels, n + 1);
	
	initial_theta = zeros(n+1, 1);
	options = optimset('GradObj', 'on', 'MaxIter', 50);
	
	for class=1:num_labels
		y_current_class = (y == class);
		theta = fmincg(@(t)(lrCostFunction(t, X, y_current_class, lambda)), initial_theta, options);
		all_theta(class, :) = theta';
	end

end
