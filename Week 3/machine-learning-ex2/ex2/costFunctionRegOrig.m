function [J, grad] = costFunctionReg(theta, X, y, lambda)

	m = length(y); % number of training examples
	n = size(X,2);	% number of features
	
	hypothesis = sigmoid(X * theta);
	
	J = 0;
	for train_set_num=1:m
		J += costOfSingleIncorrectPrediction(hypothesis(train_set_num,1), y(train_set_num,1));
	end
	J = J / m;
	reg_cost = (lambda / (2 * m)) * (theta(2:n)' * theta(2:n));
	J += reg_cost;
	
	diff = hypothesis - y;
	grad = ((diff' * X)') / m;
	grad(2:n) += (lambda / m) * theta(2:n);

	
	function [cost] = costOfSingleIncorrectPrediction(hypothesis, prediction)
		if prediction == 1
			cost = -log(hypothesis);	%penalizing the hypothesis (maximising cost) the closer to 0 it is
		elseif prediction == 0
			cost = -log(1-hypothesis);
		else
			error('Unknown category value, category value should be 0 or 1');
		end
	end
	
end
