function X_rec = recoverData(Z, U, K)
%Recovers an approximation of the original data when using the 
%projected data

	U_reduce = U(:, 1:K);
	X_rec = Z * U_reduce';

end
