function g = sigmoidGradient(z)
% Returns the gradient of the sigmoid function evaluated at z. (g'(z))
% z can be a matrix, vector or scalar

	sig = sigmoid(z);
	g = sig .* (1-sig);

end
