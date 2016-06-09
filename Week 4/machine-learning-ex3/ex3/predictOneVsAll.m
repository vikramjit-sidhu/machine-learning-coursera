function p = predictOneVsAll(all_theta, X)

% For each of the training examples in X, 
% we find the hypothesis= of each of the classifiers
% The classifier which is most confident for a particular training set is chosen

	m = size(X, 1);
	num_labels = size(all_theta, 1);

	% Add ones to the X data matrix
	X = [ones(m, 1) X];

	predictions_all = (sigmoid(all_theta * X'))';
	[max_hypothesis, p] = max(predictions_all, [], 2);	% get max prediction in each row

end
