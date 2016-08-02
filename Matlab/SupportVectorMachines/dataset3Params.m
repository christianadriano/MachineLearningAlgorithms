function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%

%Run training


%Run cross validation
sigma_vec = [0.01 0.03 0.1 0.3 1 3 10 30];
C_vec = sigma_vec;

%Sotre the outcomes from the cross-validation using all combinations of sigma_vec and C_vec
errors = zeros(8,8);

bestC=0;
bestSigma=0;
minError=1000;
error=1000;

for i=1: length(sigma_vec)
    sigma = sigma_vec(1,i);
    
    for j=1:length(C_vec)
        C = C_vec(1,j);
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        %Compute the predictions based on teh model and the inputs from the
        %cross-validation set
        predictions = svmPredict(model, Xval);
        
        %Compute how many predictions were correct
        error = mean(double(predictions ~=yval));
        
        %Save the smallest one
        if (error<minError)
           bestC = C;
           bestSigma = sigma;
           minError = error;
        end
        
%         errors(i,j) = error;
    end
end

%Another way is to store the errors in a matrix
% [minRow,rowIndex] = min(errors);
% sigma = sigma_vec(rowIndex)
% [minItem, column] = min(minRow);
% C = C_vec(column)

sigma=bestSigma;
C=bestC;


% =========================================================================

end
