function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));
%fprintf('\n initilized g in sigmoid: %f',g);


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

g = logsig(z);

%for i = 1:size(z) 
 %   g(i) = logsig(z(i));
    %g(i) = 1 / (1 + exp(-z(i)));
   % fprintf('\n g in sigmoid: %f',g);

%end

%fprintf('\n  g = logsig(z(i)) %f',logsig(z(1)));

%fprintf('\n 1 / (1 + exp(-z(i))) %f',1 / (1 + exp(-z(1))));
%fprintf('\n g in sigmoid: %f',g);

% =============================================================

end
