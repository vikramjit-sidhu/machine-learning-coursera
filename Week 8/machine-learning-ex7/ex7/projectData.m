function Z = projectData(X, U, K)
%Computes the reduced data representation when projecting only 
%on to the top k eigenvectors

	U_reduce = U(:, 1:K);
	Z = (U_reduce' * X')';

end
