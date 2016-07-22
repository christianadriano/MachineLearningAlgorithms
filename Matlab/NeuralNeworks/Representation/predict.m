function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Add ones to the X data matrix 
 a1 = [ones(m, 1) X]; 
% %a(8,3) * Theta1'(3,4)
 z2 = a1*Theta1';
% 

% %a2(8,4)
a2 = sigmoid(z2); 
% 
% 
%a2_appended(8,5)
a2_appended = [ones(size(a2,1), 1) a2];
% 
%Theta2'(5,4)
z3 = a2_appended * Theta2';

%h_probabilities(8,4) 
h_probabilities = sigmoid(z3);
% 

[maxValues,p] = max(h_probabilities, [],2)
% 
% 
% 
% 
% 
% % =========================================================================
% 
% 
% end
