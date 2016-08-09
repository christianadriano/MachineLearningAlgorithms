function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

%Obtain the centroid for that data point

%ITERATIVE SOLUTION
%For each centroid, obtain all data points
for j=1:K
    %row vector with the same number of columns of X
    totalVector = zeros(1,size(X,2));
    
    %Finds all occurrences of the centroid in the idx vector (which is the
    %vector that maps each sample to a centroid.
    [rowVector,columnVector] = find(idx==j);
  %  fprintf('j= %.f, row = %.f \n',j, rowVector);
   % size(rowVector)
    for i=1:size(rowVector,1)
        
        totalVector = totalVector + X(rowVector(i,1),:);
    end  
    centroids(j,:) = totalVector / (size(rowVector,1));
end


%SIMPLER ONE
% sel = find(idx == i) % i ranges from 1 to K
% centroids(i,:) = mean(X(sel,:))

% =============================================================


end

