function [J, grad] = smCostFunction(theta, X, Y, num_labels)
%softmax cost function
m = size(X, 1);
%y
param_size = size(X, 2);
%size(theta)
grad = zeros(size(theta));
theta = reshape(theta, param_size, num_labels-1);
%Y(:, 1)
%theta(1, :)
%theta for last K lable is always zero
theta=[theta, zeros(param_size,1)];
%size(theta)
%size(Y)
%size(X)
%Y*theta'*X
expm = exp(theta'*X');
sumexp = sum(expm);
%size(expm)
%size(sumexp)
probm = expm ./ sumexp;
h=bsxfun(@times, Y, log(probm));
%h
J = -1/m*sum(sum(h));
%size(probm)
%X
%Y
%probm
%Y-probm
%X'*(Y-probm)'
grad = -1/m*X'*(Y - probm)';
grad = grad(:, 1:end-1);
grad = grad(:);
%size(grad)

%fprintf('softmax cost function J=%f\n', J);
