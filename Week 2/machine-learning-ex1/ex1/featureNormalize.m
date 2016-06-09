function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       

% mu = mean(X);
% num_cols_X = size(X, 2);

% for col_num = 1:num_cols_X
	% X_norm(:,col_num) = X(:,col_num) - mu(1,col_num);
% end

% sigma = std(X_norm);

% for col_num = 1:num_cols_X
	% X_norm(:,col_num) = X_norm(:,col_num) / sigma(1,col_num);
% end


mu = mean(X);
sigma = std(X);
num_cols_X = size(X, 2);

for i=1:num_cols_X
	X_norm(:,i) = (X(:,i) - mu(1,i)) / sigma(1,i);	%setting all the rows, and i'th column of X_norm
end


% ============================================================

end
