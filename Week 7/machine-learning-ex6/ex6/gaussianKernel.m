function sim = gaussianKernel(x1, x2, sigma)
%Returns the gaussian kernel between x1 and x2

	% x1 and x2 should be column vectors
	x1 = x1(:); x2 = x2(:);
	
	diff = x1 - x2;
	sim = exp( -(diff' * diff) / (2*sigma^2) );
    
end
