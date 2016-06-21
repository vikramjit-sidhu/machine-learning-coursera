function idx = findClosestCentroids(X, centroids)
% Returns a vector of the centroids closest to each of the training examples
% X is an mxn matrix containing the training examples
% centroids is a Kxn matrix containing the centroids

	K = size(centroids, 1);	% Number of centroids
	m = size(X,1);	% Number of training examples
	
	idx = zeros(size(X,1), 1);

	for i=1:m
		idx(i) = computeClosestCentroidIndex(X(i,:)', centroids);
	end

	
	function index = computeClosestCentroidIndex(x, Centroids)
		minDist = inf;
		for ind = 1:size(Centroids, 1)
			distance = 	norm(x - Centroids(ind,:)');
			if distance < minDist
				minDist = distance;
				index = ind;
			end
		end
	end
end
