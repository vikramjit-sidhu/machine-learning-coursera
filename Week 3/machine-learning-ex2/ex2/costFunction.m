function [J, grad] = costFunction(theta, X, y)

	m = length(y); % number of training examples

	hypothesis = sigmoid(X * theta);

	J = 0;
	for train_set_num=1:m
		J += costOfSingleIncorrectPrediction(hypothesis(train_set_num,1), y(train_set_num,1));
	end
	J = J / m;

	diff = hypothesis - y;
	grad = ((diff' * X)') / m;
	
	%Alternate approach, non vectorized
	% n = length(theta);
	% grad = zeros(n,1);
	% for j=1:n
		% summation_term = 0;
		% for i=1:m
			% summation_term += (hypothesis(i) - y(i)) * X(i,j);
		% end
		% grad(j) = summation_term / m;
	% end
	%Alternate approach ends
	
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
