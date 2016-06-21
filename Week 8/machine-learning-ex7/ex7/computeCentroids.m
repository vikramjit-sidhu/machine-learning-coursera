function centroids = computeCentroids(X, idx, K)
% Returs the new centroids by computing the means of the 
% data points assigned to each centroid.

	[m n] = size(X);
	centroids = zeros(K, n);
	count_centroids = zeros(K, 1);
	
	for i=1:m
		centroids(idx(i), :) += X(i, :);
		count_centroids(idx(i)) += 1;
	end
	
	centroids = centroids ./ count_centroids;
end
