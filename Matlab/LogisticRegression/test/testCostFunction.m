%% Initialize

X = [ones(3,1) magic(3)];
y = [1 0 1]';
initial_theta = [-2 -1 1 2]';

% un-regularized
[j, g] = costFunction(initial_theta, X, y);

fprintf('j: %f\n', j);
fprintf('g: \n');
fprintf(' %f \n', g);

% or...
%[j g] = costFunctionReg(theta, X, y, 0)

% results
%j = 4.6832

%g =
%  0.31722
%  0.87232
%  1.64812
%  2.23787

%% regularized
clear;
X = [ones(3,1) magic(3)];
y = [1 0 1]';
theta = [-2 -1 1 2]';

[j , g] = costFunctionReg(theta, X, y, 3);
% note: also works for ex3 lrCostFunction(theta, X, y, 3)

fprintf('j: %f\n', j);
fprintf('g: \n');
fprintf(' %f \n', g);

% results
%j = 7.6832

%g =
%  0.31722
%  -0.12768
%  2.64812
%  4.23787