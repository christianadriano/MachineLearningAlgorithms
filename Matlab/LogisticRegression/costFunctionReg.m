function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


theta_modified = theta
theta_modified(1) = 0 ;
theta_squared = sum(theta_modified.^2);
theta_squared = theta_squared * lambda / (2*m);
%theta_squared = theta_squared /m;

alpha = 1;
sigmoidValue= sigmoid(X*theta);

J = ( -1* y' * log(sigmoidValue)  - (1-y)'* log(1-sigmoidValue))/m  + (theta_squared);

grad = ( X' * (sigmoidValue - y) * (alpha/ m)) + (lambda * theta_modified /m);

% =============================================================

end
