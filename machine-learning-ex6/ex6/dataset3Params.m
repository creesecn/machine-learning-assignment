function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
sigma_seqs = [[0.01, 0.03]'*10.^(0:3)](:)';
C_seqs = [[0.01, 0.03]'*10.^(0:3)](:)';
%C_seqs = [[1, 3]'*10.^(0:2)](:)';
%sigma_seqs = [0.01, 0.03, 0.1];
%C_seqs = [1, 3];
%pred_mat = zeros(length(sigma_seqs), length(C_seqs));

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
min_pred_error = 0;
%for i = 1:size(pred_mat, 1)
%    sigma_t = sigma_seqs(i);
%    for j = 1:size(pred_mat, 2)
%C_t = C_seqs(j);
for sigma_t = sigma_seqs
    for C_t = C_seqs
        fprintf("training with C=%f sigma=%f\n", C_t, sigma_t)
        model = svmTrain(X, y, C_t, @(x1, x2) gaussianKernel(x1, x2, sigma_t)); 
        pred = svmPredict(model, Xval);
        pred_error = mean(double(pred != yval));
        if (min_pred_error == 0 || pred_error < min_pred_error)
            C = C_t;
            sigma = sigma_t;
            min_pred_error = pred_error;
        end
        %pred_mat(i, j) = pred_error;
        fprintf("C=%f sigma=%f pred_error=%f\n", C_t, sigma_t, pred_error)
        fprintf("current optimal C=%f sigma=%f pred_error=%f\n", C, sigma, min_pred_error)
    end
end

fprintf("optimal C=%f sigma=%f pred_error=%f\n", C, sigma, min_pred_error)
% =========================================================================


end
