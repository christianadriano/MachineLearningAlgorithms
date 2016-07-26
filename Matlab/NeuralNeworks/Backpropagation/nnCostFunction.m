function [J grad] = nnCostFunction(nn_params, ...
    input_layer_size, ...
    hidden_layer_size, ...
    num_labels, ...
    X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
    hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
    num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%Part-1

a1 = [ones(m, 1) X]; %matrix of 5000x401
a2 = sigmoid (a1*Theta1'); %matrix of 5000x25
a2 = [ones(size(a2,1), 1) a2]; %append a column of ones to a2

a3 = sigmoid (a2*Theta2'); %matrix of 5000x10 (3x4)
%fprintf('\n ---- Sizes ---\n');
%fprintf('\n size(X) %.0f x %.0f',size(X,1),size(X,2));
%fprintf('\n size(a1) %.0f x %.0f',size(a1,1),size(a1,2));
%fprintf('\n size(a2) %.0f x %.0f',size(a2,1),size(a2,2));
%fprintf('\n size(a3) %.0f x %.0f',size(a3,1),size(a3,2));
%fprintf('\n ---- Sizes ---\n');

%Consolidate all logical vectors in a single matrix
for k =1:num_labels;
    
    y_zero_ones = (y==k);
    Y(:,k)= y_zero_ones;
    
end

%fprintf('\n size(Y) %f',size(Y));

%Remember to ignore the first column for regularization
%We do that by considering only from the second line
regularization = lambda/(2*m) * (sum(sum(Theta1(:,2:end).^2,2))  +  sum(sum(Theta2(:,2:end).^2,2)));

J = sum(sum(-Y .* log(a3)  - (1-Y) .* log(1-a3),2))/(m)+ regularization;


%Part-3
%Compute the delta errors

delta3 = a3 - Y; %5000x10 5000x10
%fprintf('\n size(delta3) %0.f x %0.f',size(delta3,1),size(delta3,2));
%fprintf('\n size(a2) %.0f x %.0f',size(a2,1),size(a2,2));

delta2 = delta3*Theta2.*a2.*(1-a2);
%fprintf('\n size(delta2) %.0f x %.0f',size(delta2,1),size(delta2,2));

%Compute gradients by using backward propagation
%partial derivative of Theta = g'(z1) = a1 .* (1-a1)

%Step-4
%Accumulate the gradients
cummulativeDelta2 =  delta3'* a2;
%fprintf('\n size(cummulativeDelta2) %.0f x %.0f',size(cummulativeDelta2,1),size(cummulativeDelta2,2));

cummulativeDelta1 = delta2(:,2:end)' * a1;
%fprintf('\n size(cummulativeDelta1) %.0f x %.0f',size(cummulativeDelta1,1),size(cummulativeDelta1,2));

%Step-5
%Compute the regularized gradients

%fprintf('\n initial size(Theta2_grad) %.f x %.f',size(Theta2_grad,1),size(Theta2_grad,2));
%Theta2_grad = cummulativeDelta2(:,2:end)/m;
%fprintf('\n size(Theta2_grad) %.f x %.f',size(Theta2_grad,1),size(Theta2_grad,2));

%Theta2_grad = [cummulativeDelta2(:,1:1) Theta2_grad];
%fprintf('\n Theta2_grad %f',Theta2_grad);

%fprintf('\n initial size(Theta1_grad) %.f x %.f',size(Theta1_grad,1),size(Theta1_grad,2));
%Theta1_grad = cummulativeDelta1(:,2:end)/m;
%fprintf('\n size(Theta1_grad) %.f x %.f',size(Theta1_grad,1),size(Theta1_grad,2));
%Theta1_grad = [cummulativeDelta1(:,1:1)  Theta1_grad];
%fprintf('\n Theta1_grad %f',Theta1_grad);

if(lambda>0)
    cummulativeDelta2 = cummulativeDelta2/m;
    Theta2_grad = (cummulativeDelta2(:,2:end) + Theta2(:,2:end)*lambda/m );
    Theta2_grad = [cummulativeDelta2(:,1:1) Theta2_grad ];

    cummulativeDelta1 = cummulativeDelta1/m;
    Theta1_grad = (cummulativeDelta1(:,2:end) + Theta1(:,2:end)*lambda/m );
    Theta1_grad = [cummulativeDelta1(:,1:1) Theta1_grad ];
else
    Theta2_grad = cummulativeDelta2/m;
    Theta1_grad = cummulativeDelta1/m;
end    
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
fprintf('\n');


end
