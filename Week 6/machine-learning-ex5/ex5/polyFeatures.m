function [X_poly] = polyFeatures(X, p)
% Maps X (1D vector) into the p-th power
% Returns a matrix of mxp

	X_poly = [X];
	for i = 2:p
		X_poly = [X_poly X.^i];
	end

end
