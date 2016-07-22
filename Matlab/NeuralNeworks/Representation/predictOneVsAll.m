function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       

%all_theta = [ones(size(all_theta, 1), 1) all_theta];

%produces a matrix that each row is an outcome for the same example
%the outcome with the largest value corresponds to the most probable class
%for instance, a matrix like the following
% 1.0   0.0  // example 1 is from category 1
% 0.1   0.7  // example 2 is from category 2
% 0.4   0.9  // example 3 is from category 2
% 0.8   0.5  / example 4 is from category 1
h_probability = sigmoid(X  * all_theta'); 

[maxProbabilites,columnIndices] = max(h_probability, [], 2); 

p=columnIndices;

fprintf('\n  size(X), %f: ', size(columnIndices));


% =========================================================================


end
