function W = randInitializeWeights(L_in, L_out)
% Randomly initialize the weights of a layer with L_in
% incoming connections and L_out outgoing connections
% W will be (L_out) x (L_in+1), because of the bias term

	epsilon = 0.12;	% range of values in W
	W = rand(L_out, L_in + 1) * (2*epsilon);
	W = W .- epsilon;

end
