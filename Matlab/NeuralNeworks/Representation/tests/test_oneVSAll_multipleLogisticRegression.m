%% TEST
%input:
X = [magic(3) ; sin(1:3); cos(1:3)];
m = size(X,1);
%X = [ones(m, 1) X];
printmat(X ,'X ');

%
y = [1; 2; 2; 1; 3];
num_labels = 3;
lambda = 0.1;
[all_theta] = oneVsAll(X, y, num_labels, lambda)

RLAB ='1 2 3 4';
CLAB = RLAB;
printmat(all_theta,'All_Theta',RLAB,CLAB) 


%% EXPECTED RESULTS
%output:
% all_theta =
%   -0.559478   0.619220  -0.550361  -0.093502
%   -5.472920  -0.471565   1.261046   0.634767
%    0.068368  -0.375582  -1.652262  -1.410138