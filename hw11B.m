%%% Marium Yousuf
%%% CSC 577
%%% Assignment 11
function hw11B()
% this function incorporates all the programming for hw11
% for CSC 577 - Intro to Computer Vision
close all
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
% img_pair11 = imread("sift/slide1.jpeg");
% imshow(img_pair11);
% datacursormode on
% img_coords = ginput(8);
% writematrix(img_coords, 'slide1_coords8.txt', 'Delimiter', 'tab');
% datacursormode off
% 
% img_pair12 = imread("sift/frame1.jpg");
% imshow(img_pair12);
% datacursormode on
% img_coords = ginput(8);
% writematrix(img_coords, 'frame1_coords8.txt', 'Delimiter', 'tab');
% datacursormode off
% 
% img_pair11 = imread("sift/slide1.jpeg");
% slide_matches = readmatrix("slide1_coords8.txt");
% row_actual_slide = cast(slide_matches(:, 2), 'uint32');
% col_actual_slide = cast(slide_matches(:, 1), 'uint32');
% imshow(img_pair11);
% hold on
% plot(col_actual_slide, row_actual_slide, 'rs', 'MarkerSize', 10);
% hold off
% 
% img_pair12 = imread("sift/frame1.jpg");
% frame_matches = readmatrix("frame1_coords8.txt");
% row_actual_frame = cast(frame_matches(:, 2), 'uint32');
% col_actual_frame = cast(frame_matches(:, 1), 'uint32');
% imshow(img_pair12);
% hold on
% plot(col_actual_frame, row_actual_frame, 'rs', 'MarkerSize', 10);
% hold off

% SLIDE/FRAME 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% img_pair21 = imread("sift/slide2.jpeg");
% imshow(img_pair21);
% datacursormode on
% img_coords = ginput(8);
% writematrix(img_coords, 'slide2_coords8.txt', 'Delimiter','tab');
% datacursormode off

% img_pair22 = imread("sift/frame2.jpg");
% imshow(img_pair22);
% datacursormode on
% img_coords = ginput(8);
% writematrix(img_coords, 'frame2_coords8.txt', 'Delimiter','tab');
% datacursormode off
% 
% img_pair21 = imread("sift/slide2.jpeg");
% slide_matches = readmatrix("slide2_coords8.txt");
% row_actual_slide = cast(slide_matches(:, 2), 'uint32');
% col_actual_slide = cast(slide_matches(:, 1), 'uint32');
% imshow(img_pair21);
% hold on
% plot(col_actual_slide, row_actual_slide, 'rs', 'MarkerSize', 10);
% hold off
% 
% img_pair22 = imread("sift/frame2.jpg");
% frame_matches = readmatrix("frame2_coords8.txt");
% row_actual_frame = cast(frame_matches(:, 2), 'uint32');
% col_actual_frame = cast(frame_matches(:, 1), 'uint32');
% imshow(img_pair22);
% hold on
% plot(col_actual_frame, row_actual_frame, 'rs', 'MarkerSize', 10);
% hold off

% SLIDE/FRAME 3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% img_pair31 = imread("sift/slide3.jpeg");
% imshow(img_pair31);
% datacursormode on
% img_coords = ginput(8);
% writematrix(img_coords, 'slide3_coords8.txt', 'Delimiter','tab');
% datacursormode off
% 
% img_pair32 = imread("sift/frame3.jpg");
% imshow(img_pair32);
% datacursormode on
% img_coords = ginput(8);
% writematrix(img_coords, 'frame3_coords8.txt', 'Delimiter','tab');
% datacursormode off
%
% img_pair31 = imread("sift/slide3.jpeg");
% slide_matches = readmatrix("slide3_coords8.txt");
% row_actual_slide = cast(slide_matches(:, 2), 'uint32');
% col_actual_slide = cast(slide_matches(:, 1), 'uint32');
% imshow(img_pair31);
% hold on
% plot(col_actual_slide, row_actual_slide, 'rs', 'MarkerSize', 10);
% hold off
%
% img_pair32 = imread("sift/frame3.jpg");
% frame_matches = readmatrix("frame3_coords8.txt");
% row_actual_frame = cast(frame_matches(:, 2), 'uint32')
% col_actual_frame = cast(frame_matches(:, 1), 'uint32')
% imshow(img_pair32);
% hold on
% plot(col_actual_frame, row_actual_frame, 'rs', 'MarkerSize', 10);
% hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%**********************
% DLT METHOD STARTS HERE TO COMPUTE HOMOGRAPHY %**********************
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%**********************
% SLIDE/FRAME PAIR 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
img_pair11 = imread("sift/slide1.jpeg");
img_pair12 = imread("sift/frame1.jpg");
slide_matches = readmatrix("mouse-click_coords/slide1_coords8.txt");
frame_matches = readmatrix("mouse-click_coords/frame1_coords8.txt");
X_4 = [slide_matches(1:2:8, :), ones(4, 1)];
Xp_4 = [frame_matches(1:2:8, :), ones(4, 1)];
n = 4;
H = DLT(X_4, Xp_4, n);

X_8 = [slide_matches, ones(size(slide_matches, 1), 1)];
Xp_8 = [frame_matches, ones(size(frame_matches, 1), 1)]
pred = H*X_8';
pred = pred';
pred = pred ./ pred(:, 3)
% rms(Xp_8 - pred, 'all') % 0.1750

Xp_plt = pred(:, 1:2);

frame_matches = readmatrix("mouse-click_coords/frame1_coords8.txt");
row_actual_frame = cast(frame_matches(:, 2), 'uint32');
col_actual_frame = cast(frame_matches(:, 1), 'uint32');
row_est_frame = cast(Xp_plt(:, 2), 'uint32'); 
col_est_frame = cast(Xp_plt(:, 1), 'uint32'); 
imshow(img_pair12);
hold on
plot(col_actual_frame, row_actual_frame, 'rs', 'MarkerSize', 7, 'LineWidth', 2);
hold on
plot(col_est_frame, row_est_frame, 'ys', 'MarkerSize', 3, 'LineWidth', 3);
hold off

% % SLIDE/FRAME PAIR 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
img_pair21 = imread("sift/slide2.jpeg");
img_pair22 = imread("sift/frame2.jpg");
slide_matches = readmatrix("mouse-click_coords/slide2_coords8.txt");
frame_matches = readmatrix("mouse-click_coords/frame2_coords8.txt");
X_4 = [slide_matches(1:2:8, :), ones(4, 1)];
Xp_4 = [frame_matches(1:2:8, :), ones(4, 1)];
n = 4;
H = DLT(X_4, Xp_4, n);

X_8 = [slide_matches, ones(size(slide_matches, 1), 1)];
Xp_8 = [frame_matches, ones(size(frame_matches, 1), 1)]
pred = H*X_8';
pred = pred';
pred = pred ./ pred(:, 3)
rms(Xp_8 - pred, 'all') % 0.1756
Xp_plt = pred(:, 1:2);
% Xp_plt = rescale(estim_Xp(:, 1:2), 1, mean([size(img_pair21, 1), size(img_pair22, 2)]));

frame_matches = readmatrix("mouse-click_coords/frame2_coords8.txt");
row_actual_frame = cast(frame_matches(:, 2), 'uint32');
col_actual_frame = cast(frame_matches(:, 1), 'uint32');
row_est_frame = cast(Xp_plt(:, 2), 'uint32'); 
col_est_frame = cast(Xp_plt(:, 1), 'uint32'); 
imshow(img_pair22);
hold on
plot(col_actual_frame, row_actual_frame, 'rs', 'MarkerSize', 7, 'LineWidth', 2);
hold on
plot(col_est_frame, row_est_frame, 'ys', 'MarkerSize', 3, 'LineWidth', 3);
hold off

% % SLIDE/FRAME PAIR 3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
img_pair31 = imread("sift/slide3.jpeg");
img_pair32 = imread("sift/frame3.jpg");
slide_matches = readmatrix("mouse-click_coords/slide3_coords8.txt");
frame_matches = readmatrix("mouse-click_coords/frame3_coords8.txt");
X_4 = [slide_matches(1:2:8, :), ones(4, 1)];
Xp_4 = [frame_matches(1:2:8, :), ones(4, 1)];
n = 4;
H = DLT(X_4, Xp_4, n);
% save('H_sf3.mat','H');

X_8 = [slide_matches, ones(size(slide_matches, 1), 1)];
Xp_8 = [frame_matches, ones(size(frame_matches, 1), 1)];
pred = H*X_8';
pred = pred';
pred = pred ./ pred(:, 3)
rms(Xp_8 - pred, 'all') % 0.1756
Xp_plt = pred(:, 1:2);
% rms(Xp_8' - H*X_8', 'all') % 0.1947

% estim_Xp = (H*X_8')'
% Xp_plt = rescale(estim_Xp(:, 1:2), 1, mean([size(img_pair31, 1), size(img_pair32, 2)]));

frame_matches = readmatrix("mouse-click_coords/frame3_coords8.txt");
row_actual_frame = cast(frame_matches(:, 2), 'uint32')
col_actual_frame = cast(frame_matches(:, 1), 'uint32')
row_est_frame = cast(Xp_plt(:, 2), 'uint32'); 
col_est_frame = cast(Xp_plt(:, 1), 'uint32'); 
imshow(img_pair32);
hold on
plot(col_actual_frame, row_actual_frame, 'rs', 'MarkerSize', 7, 'LineWidth', 2);
hold on
plot(col_est_frame, row_est_frame, 'ys', 'MarkerSize', 3, 'LineWidth', 3);
hold off
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
    [V, D] = eig(matY);
    D = diag(D);
    [~, m] = min(D);
    H_vec = V(:, m);
    H = reshape(H_vec, 3, 3)';
end