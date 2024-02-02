% Learn to write training (parameter estimation) algorithm for the Radial Basis Function Network based approximator

% Step 1: Define the input, desired output and Gaussian Radial Basis Functions(RBF)
X = 0.1:1/22:1; % Input data
Y = (1 + 0.6*sin(2*pi*X/0.7) + 0.3*sin(2*pi*X/2)); % Desired Output

c1 = 0.19; % Center of the first Gaussian RBF
c2 = 0.87; % Center of the second Gaussian RBF
r1 = 0.2; % Radius for the first RBF
r2 = 0.2; % Radius for the second RBF
F1 = exp(-(X-c1).^2/(2*r1^2)); % First Gaussian RBF 
F2 = exp(-(X-c2).^2/(2*r2^2)); % Second Gaussian RBF

% Step 2: Initialize the weights, learning rate and number of epochs
w1 = rand(1); % Weight for the first RBF
w2 = rand(1); % Weight for the second RBF
w0 = rand(1); % Bias weight
learning_rate = 0.01;
max_epochs = 1000;

% Step 3: Train the Network
for epochs = 1:max_epochs

    % Loop over each input-output pair
    for i = 1:length(X)

        % Compute the output of the output layer
        Y_pred = w1 * F1(i) + w2 * F2(i) + w0;
        
        % Compute the error
        e = Y(i) - Y_pred;
        
        % Update the weights
        w1 = w1 + learning_rate * e * F1(i);
        w2 = w2 + learning_rate * e * F2(i);
        w0 = w0 + learning_rate * e;
    end

    % Compute the RMSE
    Y_pred = w1 * F1 + w2 * F2 + w0;
    rmse = sqrt(mean((Y - Y_pred).^2));
    
    % Display the RMSE and the weights
    fprintf('Iteration %d: RMSE = %.4f, w1 = %.4f, w2 = %.4f, w0 = %.4f\n', epochs, rmse, w1, w2, w0);

    % Check if the RMSE is below a threshold
    if rmse < 0.01
        break
    end
end

% Plot the approximation and the target function
figure
plot(X, Y,'r-o', X, Y_pred,'b-o')
xlabel('X')
ylabel('Y')
legend('Target','Approximation','Location', 'Northoutside')
title('RBF Network Approximation')
grid on