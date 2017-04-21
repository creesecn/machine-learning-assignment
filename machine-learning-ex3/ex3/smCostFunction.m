function [J, grad] = smCostFunction(theta, X, Y, num_labels)
%softmax cost function
m = size(X, 1);
param_size = size(X, 2);
grad = zeros(size(theta));
theta = reshape(theta, param_size, num_labels-1);
%theta for last K lable is always zero
theta=[theta, zeros(param_size,1)];
expm = exp(theta'*X');
sumexp = sum(expm);
%dup sum row match expm to avoid automatic broadcasting warning
%sumexp = repmat(sum(expm), num_labels, 1);
%probm = expm ./ sumexp;
probm = bsxfun(@rdivide, expm, sumexp);
h = bsxfun(@times, Y, log(probm));
J = -1/m*sum(sum(h));
grad = -1/m*X'*(Y - probm)';
grad = grad(:, 1:end-1);
grad = grad(:);

%fprintf('softmax cost function J=%f\n', J);
