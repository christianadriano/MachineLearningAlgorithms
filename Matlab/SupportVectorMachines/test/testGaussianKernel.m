%% Test case for Gaussian Kernel computation
%ans =  0.45943
gaussianKernel([1 2 3], [2 4 6], 3)

%ans = 0
gaussianKernel([1 1], [100 100], 1);

%ans =1
gaussianKernel([1 1], [1 1], 1);