function p = predict(theta, X)

	m = size(X, 1); % Number of training examples
	THRESHOLD = 0.5;
	p = zeros(m, 1);

	hypothesis = sigmoid(X * theta);
	
	for index=1:m
		if hypothesis(index) >= THRESHOLD
			p(index) = 1;
		else
			p(index) = 0;
		end
	end

end
