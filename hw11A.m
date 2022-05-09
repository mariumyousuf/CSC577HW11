%%% Marium Yousuf
%%% CSC 577
%%% Assignment 11
function hw11A()
% this function incorporates all the programming for hw11
% for CSC 577 - Intro to Computer Vision
close all
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PART A - RANSAC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data = readmatrix("line_data_2");
N = size(data, 1);
X = data(:, 1);
Y = data(:, 2);

d = N*0.25; % at least 1/4 pts assumed to be close to good fit
w = 0.6;
n = 2;
p = 0.99;

k = findk(w, n, p)
bestErr = 100000;
for idx=1:k
    inl = datasample(data, 2);
    [~, ~, distParam, n_vec] = homogenousLS(inl);
    [data_diff, ~] = setdiff(data, inl, 'rows');
    for p=1:size(data_diff, 1)
        closePoints = homogenousLSmodel(data_diff, distParam, n_vec, 0.2);
        if size(closePoints, 1) > d
            % found a good model - compute line fit params for plotting
            allPoints = [closePoints; inl];
            [currErr, estY] = homogenousLS(allPoints);
            if currErr < bestErr
                X_best = allPoints(:, 1);
                bestFit = estY;
                bestErr = currErr;
            end
        else
            k = k + 1;
        end
    end
end
plot(X, Y, 'o');
axis([-1 4 -1 4]);
hold on
plot(X_best, bestFit);
disp(bestErr)
end

function k = findk(w, n, p)
    % w is an inlier ratio
    t1 = log(1-p);
    t2 = log(1-(w^n));
    k = ceil(t1/t2);
end

function nP = homogenousLSmodel(data, d_2, unit_vector_n, thresh)
    % we use homogenous LS model to compute the perpendicular distance 
    % from the origin to the line d; nP refers to nearby points
    dist = abs(data*unit_vector_n - d_2);
    bool = dist < thresh;
    nP = data.*bool;
    abs_d_i_2 = dist.*bool;
    nP(nP(:, 1) == 0, :) = [];
end

function [err, est_Y, d_2, unit_vector_n] = homogenousLS(data)
    X = data(:, 1);
    Y = data(:, 2);
    N = size(data, 1);
    x_bar = mean(X);
    y_bar = mean(Y);
    X_prime = X - x_bar;
    Y_prime = Y - y_bar;
    U_2 = [X_prime, Y_prime];
    SVD_var = U_2'*U_2;
    [V, ~] = eig(SVD_var);
    unit_vector_n = V(:, 1);
    d_2 = [x_bar; y_bar]'*unit_vector_n;
    abs_d = abs(data*unit_vector_n - d_2);
    err = sqrt(sum(abs_d.^2)/N);
    slope = -unit_vector_n(1) / unit_vector_n(2);
    intercept = d_2 / unit_vector_n(2);
    est_Y = slope*X + intercept;
end