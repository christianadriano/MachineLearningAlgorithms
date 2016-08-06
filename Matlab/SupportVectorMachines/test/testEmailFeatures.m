%% Test email features
% input
idx = [2 4 6 8 2 4 6 8]';
v = emailFeatures(idx);
v(1:10)
sum(v)
%% expected rsults
% results
% >> v(1:10)
% ans =
%    0
%    1
%    0
%    1
%    0
%    1
%    0
%    1
%    0
%    0
% 
% >> sum(v)
% ans =  4