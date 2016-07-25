%% Test the cost and gradient for backwards propagation
il = 2;              % input layer
hl = 2;              % hidden layer
nl = 4;              % number of labels
nn = [ 1:18 ] / 10;  % nn_params
X = cos([1 2 ; 3 4 ; 5 6]); %3 x 2
y = [4; 2; 3];
lambda = 4;
[J grad] = nnCostFunction(nn, il, hl, nl, X, y, lambda)

%% expected outcomes
% J = 19.474
% grad =
% 0.76614
% 0.97990
% 0.37246
% 0.49749
% 0.64174
% 0.74614
% 0.88342
% 0.56876
% 0.58467
% 0.59814
% 1.92598
% 1.94462
% 1.98965
% 2.17855
% 2.47834
% 2.50225
% 2.52644
% 2.72233