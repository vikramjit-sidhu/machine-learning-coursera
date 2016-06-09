function g = sigmoid(z)

	g = 1 + exp(-z);
	g = 1 ./ g;

end
