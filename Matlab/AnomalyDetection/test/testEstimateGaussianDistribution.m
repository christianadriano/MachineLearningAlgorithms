%% Test 1a (Estimate Gaussian Parameters):
%input:
X = sin(magic(4));
X = X(:,1:3);
[mu sigma2] = estimateGaussian(X);
fprintf('\n mu : %f', mu); 
fprintf('\n sigma2 : %f', sigma2); 


% output:
% mu =
%   -0.3978779  0.3892253  -0.0080072
% sigma2 =
%    0.27795    0.65844   0.20414

%% ---------------------------------------
% Test 2a (Select threshold):
%input:
[epsilon F1] = selectThreshold([1 0 0 1 1]', [0.1 0.2 0.3 0.4 0.5]')
% output:
% epsilon =  0.40040
% F1 =  0.57143