function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
    num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
    num_users, num_features);


% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the
%                     partial derivatives w.r.t. to each element of Theta

%Step-1 Computing the predicted movie ratings for all users using the product of X and Theta.

%P PredictedMovie ratings
P = X * Theta';

%fprintf('\n Size P: %.f,%.f\n', size(P,1),size(P,2));


%Step-2 Computing the movie rating error by subtracting Y from the predicted ratings.
Rating_error = (P - Y);
Error_factor = Rating_error.* R;
J = sum(sum(Error_factor.^2))/2;

CostRegularization = lambda*sum(sum(X.^2))/2 + lambda*sum(sum(Theta'.^2))/2;

J = J + CostRegularization;

%The following also works
%J =  sum(Error_factor(R==1).^2) / 2 ;


%Step-3
% for i=1:num_movies
%     X_grad(i,:) = Error_factor(i,:) * Theta
% end
%
% for j=1:num_users
%     Theta_grad(j,:) = X' * Error_factor(:,j);
% end

%Step-4 Add regularization


for i=1:num_movies
    idx = find(R(i, :)==1);
    Thetatemp = Theta(idx, :); %do not consider input from users who rated the ith movie
    Ytemp = Y(i,idx);
    RegularizedGradient =  X(i,:)*lambda;
    X_grad(i,:) =(X(i,:)*Thetatemp' - Ytemp) * Thetatemp + RegularizedGradient;
end



for j=1:num_users
    idx = find(R(:, j)==1); %for jth user, only movies that she rated
    Xtemp = X(idx,: ); %do not consider movies that the jth user did not rate.
    Ytemp = Y(idx,j);
    RegularizedGradient = Theta(j,:)*lambda;
    Theta_grad(j,:) =  Xtemp' * (Xtemp * Theta(j,:)' - Ytemp) + RegularizedGradient';
end








% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
