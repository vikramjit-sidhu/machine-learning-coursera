function [C, sigma] = dataset3Params(X, y, Xval, yval)
% Finds the value of C and sigma which will minimize the error 
% in the cross-validation set.
% The values of C and sigma are chosen from a training set.

	possible_values = [0.01 0.03 0.1 0.3 1 3 10 30]';
	min_error = Inf;
	
	for index_C = 1:size(possible_values)
		poss_C = possible_values(index_C);
		for index_sig = 1:size(possible_values)
			poss_sig = possible_values(index_sig);
			
			predictions = predictSVMValues(X, y, poss_C, poss_sig, Xval);
			error = findErrorInPredictions(predictions, yval);
			if error < min_error
				C = poss_C;
				sigma = poss_sig;
				min_error = error;
			end
		end
	end
	
	
	
	function pred = predictSVMValues(X, y, C, sigma, Xval)
		model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
		pred = svmPredict(model, Xval);
	end
	
	
	function predError = findErrorInPredictions(pred, y)
		predError = mean(double(pred != y));
	end
end
