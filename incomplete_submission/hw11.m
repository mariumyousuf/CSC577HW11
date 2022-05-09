%%% Marium Yousuf
%%% CSC 577
%%% Assignment 11
function hw11()
% this function incorporates all the programming for hw11
% for CSC 577 - Intro to Computer Vision
close all
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PART A - RANSAC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% data = readmatrix("line_data_2");
% N = size(data, 1);
% X = data(:, 1);
% Y = data(:, 2);
% 
% d = N*0.25; % at least 1/4 pts assumed to be close to good fit
% w = 0.6;
% n = 2;
% p = 0.99;
% 
% k = findk(w, n, p)
% bestErr = 100000;
% for idx=1:k
%     inl = datasample(data, 2);
%     [~, ~, distParam, n_vec] = homogenousLS(inl);
%     [data_diff, ~] = setdiff(data, inl, 'rows');
%     for p=1:size(data_diff, 1)
%         closePoints = homogenousLSmodel(data_diff, distParam, n_vec, 0.2);
%         if size(closePoints, 1) > d
%             % found a good model - compute line fit params for plotting
%             allPoints = [closePoints; inl];
%             [currErr, estY] = homogenousLS(allPoints);
%             if currErr < bestErr
%                 X_best = allPoints(:, 1);
%                 bestFit = estY;
%                 bestErr = currErr;
%             end
%         else
%             k = k + 1;
%         end
%     end
% end
% plot(X, Y, 'o');
% axis([-1 4 -1 4]);
% hold on
% plot(X_best, bestFit);
% disp(bestErr)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PART B - Homography
%
% 1) determine 4 good matches for an image pair (8 pts)
%       - saved as files slide1_coords.txt, frame1_coords.txt
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%% Part B - (1) %%%%%%%%%%%%%
% err = [];
% num_samples = 10;
% num_pairs = 4;
% for s=1:num_samples
%     pairs_A = rand(num_pairs, 2);
%     pairs_E = rand(num_pairs, 2);
%     X = [pairs_A, ones(size(pairs_A, 1), 1)];
%     Xp = [pairs_E, ones(size(pairs_E, 1), 1)];
%     H = DLT(X, Xp, num_pairs);
%     err = [err, abs(Xp' - H*X')];
% end
% rms(err, "all") % 0.3108, 0.3235, 0.3267

% %%%%%%%%%%%%% Part B - (2) %%%%%%%%%%%%%
% ALL COMMENTED OUT PARTS ARE MOUSE-CLICK POINT COLLECTIONS
% FOR MARKER VISUALIZATIONS
% SLIDE/FRAME 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% img_pair11 = imread("a9_data/slide1.jpeg");
% imshow(img_pair11);
% datacursormode on
% img_coords = ginput(8);
% writematrix(img_coords, 'slide1_coords8.txt', 'Delimiter', 'tab');
% datacursormode off
% 
% img_pair12 = imread("a9_data/frame1.jpg");
% imshow(img_pair12);
% datacursormode on
% img_coords = ginput(8);
% writematrix(img_coords, 'frame1_coords8.txt', 'Delimiter', 'tab');
% datacursormode off
% 
% img_pair11 = imread("a9_data/slide1.jpeg");
% slide_matches = readmatrix("slide1_coords8.txt");
% row_actual_slide = cast(slide_matches(:, 2), 'uint32');
% col_actual_slide = cast(slide_matches(:, 1), 'uint32');
% imshow(img_pair11);
% hold on
% plot(col_actual_slide, row_actual_slide, 'rs', 'MarkerSize', 10);
% hold off
% 
% img_pair12 = imread("a9_data/frame1.jpg");
% frame_matches = readmatrix("frame1_coords8.txt");
% row_actual_frame = cast(frame_matches(:, 2), 'uint32');
% col_actual_frame = cast(frame_matches(:, 1), 'uint32');
% imshow(img_pair12);
% hold on
% plot(col_actual_frame, row_actual_frame, 'rs', 'MarkerSize', 10);
% hold off

% SLIDE/FRAME 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% img_pair21 = imread("a9_data/slide2.jpeg");
% imshow(img_pair21);
% datacursormode on
% img_coords = ginput(8);
% writematrix(img_coords, 'slide2_coords8.txt', 'Delimiter','tab');
% datacursormode off

% img_pair22 = imread("a9_data/frame2.jpg");
% imshow(img_pair22);
% datacursormode on
% img_coords = ginput(8);
% writematrix(img_coords, 'frame2_coords8.txt', 'Delimiter','tab');
% datacursormode off
% 
% img_pair21 = imread("a9_data/slide2.jpeg");
% slide_matches = readmatrix("slide2_coords8.txt");
% row_actual_slide = cast(slide_matches(:, 2), 'uint32');
% col_actual_slide = cast(slide_matches(:, 1), 'uint32');
% imshow(img_pair21);
% hold on
% plot(col_actual_slide, row_actual_slide, 'rs', 'MarkerSize', 10);
% hold off
% 
% img_pair22 = imread("a9_data/frame2.jpg");
% frame_matches = readmatrix("frame2_coords8.txt");
% row_actual_frame = cast(frame_matches(:, 2), 'uint32');
% col_actual_frame = cast(frame_matches(:, 1), 'uint32');
% imshow(img_pair22);
% hold on
% plot(col_actual_frame, row_actual_frame, 'rs', 'MarkerSize', 10);
% hold off

% SLIDE/FRAME 3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% img_pair31 = imread("a9_data/slide3.jpeg");
% imshow(img_pair31);
% datacursormode on
% img_coords = ginput(8);
% writematrix(img_coords, 'slide3_coords8.txt', 'Delimiter','tab');
% datacursormode off
% 
% img_pair32 = imread("a9_data/frame3.jpg");
% imshow(img_pair32);
% datacursormode on
% img_coords = ginput(8);
% writematrix(img_coords, 'frame3_coords8.txt', 'Delimiter','tab');
% datacursormode off
%
% img_pair31 = imread("a9_data/slide3.jpeg");
% slide_matches = readmatrix("slide3_coords8.txt");
% row_actual_slide = cast(slide_matches(:, 2), 'uint32');
% col_actual_slide = cast(slide_matches(:, 1), 'uint32');
% imshow(img_pair31);
% hold on
% plot(col_actual_slide, row_actual_slide, 'rs', 'MarkerSize', 10);
% hold off
%
% img_pair32 = imread("a9_data/frame3.jpg");
% frame_matches = readmatrix("frame3_coords8.txt");
% row_actual_frame = cast(frame_matches(:, 2), 'uint32')
% col_actual_frame = cast(frame_matches(:, 1), 'uint32')
% imshow(img_pair32);
% hold on
% plot(col_actual_frame, row_actual_frame, 'rs', 'MarkerSize', 10);
% hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DLT METHOD STARTS HERE TO COMPUTE HOMOGRAPHY %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % SLIDE/FRAME PAIR 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% img_pair11 = imread("a9_data/slide1.jpeg");
% img_pair12 = imread("a9_data/frame1.jpg");
% slide_matches = readmatrix("slide1_coords8.txt");
% frame_matches = readmatrix("frame1_coords8.txt");
% slide_matches = rescale(slide_matches);
% frame_matches = rescale(frame_matches);
% X_4 = [slide_matches(1:2:8, :), ones(4, 1)];
% Xp_4 = [frame_matches(1:2:8, :), ones(4, 1)];
% n = 4;
% H = DLT(X_4, Xp_4, n);
% save('H_sf1.mat', 'H');
% 
% X_8 = [slide_matches, ones(size(slide_matches, 1), 1)];
% Xp_8 = [frame_matches, ones(size(frame_matches, 1), 1)];
% rms(Xp_8' - H*X_8', 'all') % 0.1750
% 
% estim_Xp = (H*X_8')'
% Xp_plt = rescale(estim_Xp(:, 1:2), 1, mean([size(img_pair12, 1), size(img_pair12, 2)]));
% 
% frame_matches = readmatrix("frame1_coords8.txt");
% row_actual_frame = cast(frame_matches(:, 2), 'uint32')
% col_actual_frame = cast(frame_matches(:, 1), 'uint32')
% row_est_frame = cast(Xp_plt(:, 2), 'uint32'); 
% col_est_frame = cast(Xp_plt(:, 1), 'uint32'); 
% imshow(img_pair12);
% hold on
% plot(col_actual_frame, row_actual_frame, 'rs', 'MarkerSize', 10);
% hold on
% plot(col_est_frame, row_est_frame, 'ys', 'MarkerSize', 10);
% hold off

% % SLIDE/FRAME PAIR 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% img_pair21 = imread("a9_data/slide2.jpeg");
% img_pair22 = imread("a9_data/frame2.jpg");
% slide_matches = readmatrix("slide2_coords8.txt");
% frame_matches = readmatrix("frame2_coords8.txt");
% slide_matches = rescale(slide_matches);
% frame_matches = rescale(frame_matches);
% X_4 = [slide_matches(1:2:8, :), ones(4, 1)];
% Xp_4 = [frame_matches(1:2:8, :), ones(4, 1)];
% n = 4;
% H = DLT(X_4, Xp_4, n);
% save('H_sf2.mat','H');
% 
% X_8 = [slide_matches, ones(size(slide_matches, 1), 1)];
% Xp_8 = [frame_matches, ones(size(frame_matches, 1), 1)];
% rms(Xp_8' - H*X_8', 'all') % 0.1756
% 
% estim_Xp = (H*X_8')'
% Xp_plt = rescale(estim_Xp(:, 1:2), 1, mean([size(img_pair21, 1), size(img_pair22, 2)]));
% 
% frame_matches = readmatrix("frame2_coords8.txt");
% row_actual_frame = cast(frame_matches(:, 2), 'uint32')
% col_actual_frame = cast(frame_matches(:, 1), 'uint32')
% row_est_frame = cast(Xp_plt(:, 2), 'uint32'); 
% col_est_frame = cast(Xp_plt(:, 1), 'uint32'); 
% imshow(img_pair22);
% hold on
% plot(col_actual_frame, row_actual_frame, 'rs', 'MarkerSize', 10);
% hold on
% plot(col_est_frame, row_est_frame, 'ys', 'MarkerSize', 10);
% hold off

% % SLIDE/FRAME PAIR 3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% img_pair31 = imread("a9_data/slide3.jpeg");
% img_pair32 = imread("a9_data/frame3.jpg");
% slide_matches = readmatrix("slide3_coords8.txt");
% frame_matches = readmatrix("frame3_coords8.txt");
% slide_matches = rescale(slide_matches);
% frame_matches = rescale(frame_matches);
% X_4 = [slide_matches(1:2:8, :), ones(4, 1)];
% Xp_4 = [frame_matches(1:2:8, :), ones(4, 1)];
% n = 4;
% H = DLT(X_4, Xp_4, n);
% save('H_sf3.mat','H');
% 
% X_8 = [slide_matches, ones(size(slide_matches, 1), 1)];
% Xp_8 = [frame_matches, ones(size(frame_matches, 1), 1)];
% rms(Xp_8' - H*X_8', 'all') % 0.1947
% 
% estim_Xp = (H*X_8')'
% Xp_plt = rescale(estim_Xp(:, 1:2), 1, mean([size(img_pair31, 1), size(img_pair32, 2)]));
% 
% frame_matches = readmatrix("frame3_coords8.txt");
% row_actual_frame = cast(frame_matches(:, 2), 'uint32')
% col_actual_frame = cast(frame_matches(:, 1), 'uint32')
% row_est_frame = cast(Xp_plt(:, 2), 'uint32'); 
% col_est_frame = cast(Xp_plt(:, 1), 'uint32'); 
% imshow(img_pair32);
% hold on
% plot(col_actual_frame, row_actual_frame, 'rs', 'MarkerSize', 10);
% hold on
% plot(col_est_frame, row_est_frame, 'ys', 'MarkerSize', 10);
% hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PART C - Homography with RANSAC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data = imread("a9_data/slide3.jpeg");
model = imread("a9_data/frame3.jpg");
H1 = matfile('H_sf3.mat');
baseH = H1.H;
n = 4;
w = 0.4;
n = 4;
p = 0.99;
k = findk(w, n, p);
err_thresh = 50; 
N = 95;

bestFit = [];
bestErr = 10000;

for itr=1:k
    % maybeInliers
    rows_s = [];
    cols_s = [];
    for i=1:n
        rows_s = [rows_s, ceil(rand * size(data, 1))];
        cols_s = [cols_s, ceil(rand * size(data, 2))];
    end
    maybeInl = [rows_s', cols_s', ones(n, 1)];
    % maybeModels
    maybeModels = (baseH*maybeInl')';
    updatedH = DLT(maybeInl, maybeModels, n);
    % alsoInliers
    alsoInliers = [];
    fit = [];
    % for points not in inliers
    for r=1:10:size(data, 1)
        for c=1:10:size(data, 2)
            % check for duplicates
            if ismember(r, maybeInl(:, 1)) & ismember(c, maybeInl(:, 2))
                continue
            % compute frame coordinates that are within the threshold
            else
                X_itr = [r, c, 1];
                Xp_est_itr = (updatedH*X_itr')';
                rows_f = ceil(rand * size(model, 1));
                cols_f = ceil(rand * size(model, 2));
                Xp = [rows_f, cols_f, 1];
                rms(Xp_est_itr - Xp, "all")
                if rms(Xp_est_itr(:, 1:2) - Xp(:, 1:2), "all") < err_thresh
                    alsoInliers = [alsoInliers; X_itr];
                    fit = [fit; (updatedH*X_itr')'];
                    size(alsoInliers, 1);
                    if size(alsoInliers, 1) > N
                        allInliers = [maybeInl; alsoInliers];
                        bestFit = (updatedH*allInliers')';
                        updatedH = DLT(allInliers, bestFit, size(allInliers, 1));
                    end
                end
            end
        end
    end
end

slide_pts_plt = allInliers(:, 1:2);
frame_pts_plt = rescale(bestFit(:, 1:2), 1, mean([size(model, 1), size(model, 2)]));

% p1 = [slide_pts_plt(:, 1); frame_pts_plt(:, 1)+slide_pts_plt(:, 1)];
% p2 = [slide_pts_plt(:, 2); frame_pts_plt(:, 2)];

% vis1 = cat(1,data,model);
% new_img = draw_segment(vis1, slide_pts_plt, frame_pts_plt, 0, 0, 0, 255);
% imshow(new_img)

row_actual_slide = cast(slide_pts_plt(:, 2), 'uint32');
col_actual_slide = cast(slide_pts_plt(:, 1), 'uint32');
C1S1 = figure();
imshow(data);
hold on
plot(row_actual_slide, col_actual_slide, 'rs', 'MarkerSize', 10);
hold off
saveas(C1S1, "C1S3.jpg");

row_actual_frame = cast(frame_pts_plt(:, 2), 'uint32');
col_actual_frame = cast(frame_pts_plt(:, 1), 'uint32');
C1F1 = figure();
imshow(model);
hold on
plot(col_actual_frame, row_actual_frame, 'rs', 'MarkerSize', 10);
hold off
saveas(C1F1, "C1F3.jpg");
end

function new_img = draw_segment(img, p2, p1, width, r, g, b)

   [ i j ] = local_bresenham ( p1(1), p1(2), p2(1), p2(2) );
   new_img = draw_points(img, [ i j ], width, r, g, b);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function new_img = draw_points (img, points, box_size, r, g, b)
   new_img = img;

   count = size(points, 1);

   for i = 1:count
       new_img = local_draw_box(new_img, points(i,1), points(i,2), box_size, r, g, b);
   end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This should be the same code as draw_box.m, but for teaching (and that is the
% main reason or Matlab), it is easier to reduce the number of dependencies.
%
function new_img = local_draw_box (img, ci, cj, box_size, r, g, b)
    new_img = img;

    [ m n d ] = size(img);

    i_min = max(1, round(ci) - box_size);
    i_max = min(m, round(ci) + box_size);

    j_min = max(1, round(cj) - box_size);
    j_max = min(n, round(cj) + box_size);

    for i =  i_min:i_max
        for j =  j_min:j_max
            new_img(i, j, 1) = r;
            new_img(i, j, 2) = g;
            new_img(i, j, 3) = b;
        end
    end
end


% Nice code from the web provided by Aaron Wetzler. 
%
% This should be the same as bresenham.m, but copied to be self contained.

function [x y] = local_bresenham(x1,y1,x2,y2)

%Matlab optmized version of Bresenham line algorithm. No loops.
%Format:
%               [x y]=bham(x1,y1,x2,y2)
%
%Input:
%               (x1,y1): Start position
%               (x2,y2): End position
%
%Output:
%               x y: the line coordinates from (x1,y1) to (x2,y2)
%
%Usage example:
%               [x y]=bham(1,1, 10,-5);
%               plot(x,y,'or');
x1=round(x1); x2=round(x2);
y1=round(y1); y2=round(y2);
dx=abs(x2-x1);
dy=abs(y2-y1);
steep=abs(dy)>abs(dx);
if steep t=dx;dx=dy;dy=t; end

%The main algorithm goes here.
if dy==0 
    q=zeros(dx+1,1);
else
    q=[0;diff(mod([floor(dx/2):-dy:-dy*dx+floor(dx/2)]',dx))>=0];
end

%and ends here.

if steep
    if y1<=y2 y=[y1:y2]'; else y=[y1:-1:y2]'; end
    if x1<=x2 x=x1+cumsum(q);else x=x1-cumsum(q); end
else
    if x1<=x2 x=[x1:x2]'; else x=[x1:-1:x2]'; end
    if y1<=y2 y=y1+cumsum(q);else y=y1-cumsum(q); end
end

end


function [err, Xp_est] = sample_matches(X_init, Xp_init, baseH)
    Xp_est = (baseH*X_init')'
    err = rms(Xp_init - Xp_est, "all");
end

function new_Xp = frame_corresp(frame, s_coords, H, num_matches, thresh)
    Xp = (H*s_coords')'
    rows_f = [];
    cols_f = [];
    for i=1:num_matches
        rows_f = [rows_f, ceil(rand * size(frame, 1))];
        cols_f = [cols_f, ceil(rand * size(frame, 2))];
    end
    Xp = [rows_f', cols_f', ones(num_matches, 1)];

    if rms(Xp_itr - Xp, "all") < thresh
        new_Xp = Xp;
    else
        new_Xp = [];
    end
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

function H = DLT(X, Xp, n)
    Xp_x = Xp(:, 1);
    Xp_y = Xp(:, 2);
    DLT_mx = [];
    for i=1:n
        mat = [zeros(1, 3), -X(i, :), Xp_y(i).*X(i, :);
            X(i, :), zeros(1, 3), -Xp_x(i).*X(i, :);
            -Xp_y(i).*X(i, :), Xp_x(i).*X(i, :), zeros(1, 3)];
        DLT_mx = [DLT_mx; mat];
    end
    matY = DLT_mx'*DLT_mx;
    [V, ~] = eig(matY);
    H_vec = V(:, 1);
    H = reshape(abs(H_vec), [3, 3])';
end

