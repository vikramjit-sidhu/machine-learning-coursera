function plotData(X, y)

	figure; hold on;

	positive_indexes = find(y == 1); 
	negative_indexes = find(y == 0);

	% green and + for positive ; red and o for negative
	plot(X(positive_indexes, 1), X(positive_indexes, 2), 'g+', 'markersize', 8);
	plot(X(negative_indexes, 1), X(negative_indexes, 2), 'ro', 'markersize', 8);

	hold off;

end
