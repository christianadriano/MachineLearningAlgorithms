%% test validation curves
clear;clc;clear all;
X = [1 2 ; 1 3 ; 1 4 ; 1 5]
y = [7 6 5 4]'
Xval = [1 7 ; 1 -2]
yval = [2 12]'
[lambda_vec, error_train, error_val,error_train_matrix] = validationCurve(X,y,Xval,yval )

%% results:
% lambda_vec =
%     0.00000
%     0.00100
%     0.00300
%     0.01000
%     0.03000
%     0.10000
%     0.30000
%     1.00000
%     3.00000
%    10.00000
% 
% error_train =
% 
%    0.00000
%    0.00000
%    0.00000
%    0.00000
%    0.00002
%    0.00024
%    0.00200
%    0.01736
%    0.08789
%    0.27778
% 
% error_val =
% 
%    0.25000
%    0.25055
%    0.25165
%    0.25553
%    0.26678
%    0.30801
%    0.43970
%    1.00347
%    2.77539
%    6.80556