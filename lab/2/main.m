% Load the dataset
data = readtable('breast_cancer_data.csv'); % Load your dataset here
X = data{:, 1:end-1}; % Features
y = data{:, end}; % Labels

% Normalize the data
X = zscore(X);

% Split the dataset into training and test sets
cv = cvpartition(size(X, 1), 'HoldOut', 0.2);
X_train = X(training(cv), :);
y_train = y(training(cv), :);
X_test = X(test(cv), :);
y_test = y(test(cv), :);

% Hyperparameters
input_size = size(X_train, 2);
lr = 0.001;
epochs = 50;

% Initialize weights and threshold
w = rand(input_size, 1);
threshold = rand();

% Train the Perceptron
for epoch = 1:epochs
    y_pred_train = forward(X_train, w, threshold);
    train_loss = compute_loss(y_train, y_pred_train);
    
    % Update weights and threshold
    [w, threshold] = backward(X_train, y_train, y_pred_train, w, threshold, lr);
end

% Predict on the test set
y_pred1 = predict(X_test, w, threshold);
accuracy = mean(y_pred1 == y_test);
fprintf('Accuracy (lr:0.001): %.2f%%\n', accuracy * 100);

% Function to compute forward pass
function out = forward(x, w, threshold)
    linear_output = x * w + threshold;
    out = sigmoid(linear_output);
end

% Function to compute sigmoid
function out = sigmoid(x)
    out = 1 ./ (1 + exp(-x));
end

% Function to compute loss
function loss = compute_loss(y_true, y_pred)
    loss = 0.5 * mean((y_pred - y_true).^2);
end

% Function to perform backward pass and update weights
function [w, threshold] = backward(x, y_true, y_pred, w, threshold, lr)
    error = y_pred - y_true;
    d_pred = error .* (y_pred .* (1 - y_pred));
    w = w - lr * (x' * d_pred);
    threshold = threshold - lr * sum(d_pred);
end

% Function to make predictions
function y_pred = predict(x, w, threshold)
    y_pred = forward(x, w, threshold) > 0.5;
end
