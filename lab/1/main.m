% Input the given matrix and calculate the result
matrix = [1, 0, sin(pi/4); 0, 1, sin(pi/2); 1, 0, 1];
output1 = sigmatrix(matrix,3,3);
output2 = tanhmatrix(matrix,3,3);
disp(['Result: ', num2str(output1)]);
disp(['Result: ', num2str(output2)]);

% Sigmoid function
function y = sigmoid(x)
    y = 2 / (1 + exp(-x)) - 1;
end

% tanh function
function y = tanh(x)
    y = (exp(x) - exp(-x))/(exp(x) + exp(-x));
end

% Derivative of sigmoid function
function y_prime = sigmoid_derivative(x)
    sig = sigmoid(x);
    y_prime = (1 - sig^2)/2;
end


% Function to process a matrix
function [res1, res2] = sigmatrix(matrix, rows, cols)
    res1 = 0;
    res2 = 0;
    % Loop over each element in the matrix
    for i = 1:rows
        for j = 1:cols
            res1 = res1 + sigmoid(matrix(i,j));
            res2 = res2 + sigmoid_derivative(matrix(i,j));
        end
    end
end

% Function to process a matrix
function res1 = tanhmatrix(matrix, rows, cols)
    res1 = 0;
    % Loop over each element in the matrix
    for i = 1:rows
        for j = 1:cols
            res1 = res1 + tanh(matrix(i,j));
        end
    end
end