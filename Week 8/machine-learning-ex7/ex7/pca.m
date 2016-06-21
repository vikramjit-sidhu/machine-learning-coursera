function [U, S] = pca(X)
%PCA Run principal component analysis on the dataset X

	[m, n] = size(X);

	Sigma = (1/m) * X' * X;
	[U, S, V] = svd(Sigma);

end
