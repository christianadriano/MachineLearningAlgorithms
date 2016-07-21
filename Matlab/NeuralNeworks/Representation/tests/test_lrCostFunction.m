%% 
% input
theta = [-2; -1; 1; 2];
X = [ones(5,1) reshape(1:15,5,3)/10];
y = [1;0;1;0;1] >= 0.5;       % creates a logical array
lambda = 3;
[J grad] = lrCostFunction(theta, X, y, lambda)

fprintf('\n cost J %f', J)
fprintf('\n grad %f', grad)

% results
% J =  2.5348
% grad =
% 
%    0.14656
%   -0.54856
%    0.72472
%    1.39800