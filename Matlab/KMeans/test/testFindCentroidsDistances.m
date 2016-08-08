%% Simple test centroid distances
%

X = magic(8);
X = X(:, 2:4);
centroids = magic(4);
centroids = centroids(:,2:4);
findClosestCentroids(X, centroids)

% results
% ans =
%   1
%   4
%   4
%   2
%   4
%   3
%   3
%   4