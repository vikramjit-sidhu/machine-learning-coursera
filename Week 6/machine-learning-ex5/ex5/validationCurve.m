function [lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval)
% Generate the train and validation errors needed to
% plot a validation curve that we can use to select lambda

	% Selected values of lambda
	lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

	error_train = zeros(length(lambda_vec), 1);
	error_val = zeros(length(lambda_vec), 1);

	for index = 1:size(lambda_vec)
		lambda = lambda_vec(index);
		theta = trainLinearReg(X, y, lambda);
		error_train(index) = linearRegCostFunction(X, y, theta, 0);
		error_val(index) = linearRegCostFunction(Xval, yval, theta, 0);
	end
end
