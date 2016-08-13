%% Test reshape (i.e., apply the transformation to restore the original data set)
Q = reshape([1:15],5,3)
recoverData(Q, magic(5), 3)

% ans =
%    172   130   183   291   394
%    214   165   206   332   448
%    256   200   229   373   502
%    298   235   252   414   556
%    340   270   275   455   610