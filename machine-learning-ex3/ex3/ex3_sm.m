%% Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all

%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the
%  linear exercise. You will need to complete the following functions
%  in this exericse:
%
%     lrCostFunction.m (logistic regression cost function)
%     oneVsAll.m
%     predictOneVsAll.m
%     predict.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this part of the exercise
input_layer_size  = 400;  % 20x20 Input Images of Digits
num_labels = 10;          % 10 labels, from 1 to 10
                          % (note that we have mapped "0" to label 10)

%% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset.
%  You will be working with a dataset that contains handwritten digits.
%

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('ex3data1.mat'); % training data stored in arrays X, y
m = size(X, 1);
%X = [ones(m, 1) X];

%% ============ Part 2a: Vectorize Logistic Regression ============
%  In this part of the exercise, you will reuse your logistic regression
%  code from the last exercise. You task here is to make sure that your
%  regularized logistic regression implementation is vectorized. After
%  that, you will implement one-vs-all classification for the handwritten
%  digit dataset.
%

% Test case for lrCostFunction
fprintf('\nTesting smCostFunction()\n');

%theta = rand(n, num_labels-1)*0.001;
%theta = theta(:);
theta_t = [-2; -1; 1; 2];
X_t = [ones(5,1) reshape(1:15,5,3)/10];
y_t = ([1;0;1;0;1] >= 0.5);
costFunc = @(t)(lrCostFunction(t, X_t, y_t, 0)); 
[J grad] = costFunc(theta_t);
numgrad = computeNumericalGradient(costFunc, theta_t);
disp([numgrad grad])


theta_t = theta_t(:);
costFunc = @(t)(smCostFunction(t, X_t, y_t+1, 2)); 
[J, grad] = costFunc(theta_t);
numgrad = computeNumericalGradient(costFunc, theta_t);
% Visually examine the two gradient computations.  The two columns
% you get should be very similar. 
disp([numgrad grad]);
%fprintf('grad size %d\n', size(grad));

X_train = [ones(m, 1) X];
n = size(X_train, 2);

initial_theta= rand(n, num_labels-1)*0.001;
options = optimset('MaxIter', 200);
theta = fmincg (@(t)(smCostFunction(t, X_train, y, num_labels)), initial_theta(:), options);
all_theta = reshape(theta, n, num_labels-1);
pred = predictOneVsAll(all_theta', X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

