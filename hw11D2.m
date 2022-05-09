%%% Marium Yousuf, Kayla Bennett
%%% CSC 577
%%% Assignment 11
function hw11D2()
% this function incorporates all the programming for hw11
% for CSC 577 - Intro to Computer Vision
close all
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PART D2 - SIFT, Homography, and RANSAC
% IMAGE 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
image_11 = imread("r2d2_1.jpg");
image_11 = image_11(1:end-1, :, :);
image_12 = imread("r2d2_2.jpg");

sift_11 = table2array(readtable("r2d2_1.sift", FileType="delimitedtext"));
sift_12 = table2array(readtable("r2d2_2.sift", FileType="delimitedtext"));
sift11 = sift_11(1:30:size(sift_11, 1), :);
sift12 = sift_12(1:30:size(sift_12, 1), :);
pixel_map = slideMatchesToFrames(sift11, sift12);
samples = pixel_map(1:15:size(pixel_map), :);

n = 4; % number of data points for the model (need 4 for homography)
w = 0.6; % inlier ratio
p = 0.99; % success probability
k = findk(w, n, p); % number of iterations
angleColNum = 10; % column number for Euclidean Distance between the matches
distColNum = 9; % column number for the angle between the match vectors

t = mean(pixel_map(:, distColNum));
N = 150;
[bestFit, bestErr] = SIFTRANSAC(pixel_map, n, k, t, N, distColNum);
bestHomography = bestFit;
homogSlidePoints = [samples(:, 5:6), ones(size(samples, 1), 1)];
estimateFramePoints = (bestHomography*homogSlidePoints')';
estimateFramePoints = estimateFramePoints ./ estimateFramePoints(:, 3);
size(estimateFramePoints)
disp(bestErr)

% stackedImgs = cat(2, image_11, image_12);
% topImgLen = size(image_11, 2);
% estimateFramePoints(:, 1) = estimateFramePoints(:,1) + topImgLen;
% for i=1:size(homogSlidePoints, 1)
%     stackedImgs = draw_segment(stackedImgs, homogSlidePoints(i, 2:-1:1), estimateFramePoints(i, 2:-1:1), 0, 0, 255, 255);
% end
% imshow(stackedImgs)

% image_11 = rescale(image_11);
% image_12 = rescale(image_12);
% height_1 = size(image_11, 1);
% width_1 = size(image_11, 1);
% corners = [[0, 0, 1]; [width_1, 0, 1]; [width_1, height_1, 1]; [0, height_1, 1]]
% corners_new = [];
% for i=1:size(corners, 1)
%     corners_new = [corners_new, bestFit*corners(i, :)'];
% end
% corners_new% = corners_new'
% x_news = corners_new(1, :) / corners_new(3, :);
% y_news = corners_new(2, :) / corners_new(3, :);
% y_min = min(y_news)
% x_min = min(x_news)
% 
% panoramaView = imref2d([height width], xLimits, yLimits);
% translation_mat = [[1, 0, -x_min]; [0, 1, -y_min]; [0, 0, 1]];
% H = translation_mat*bestFit

image_11 = rescale(image_11);
image_12 = rescale(image_12);
warpedImage = imwarp(image_11, projective2d(bestFit), );
imshow(warpedImage)
% , panoramaView);
%
% x_max = max(x_news)
% y_max = max(y_news)
% 
% width  = round(x_max - x_min)
% height = round(y_max - y_min)
% xLimits = [x_min x_max];
% yLimits = [y_min y_max];
% panoramaView = imref2d([height width], xLimits, yLimits);
% im_out = imwarp(image_11, projective2d(H'), 'nearest');
% imshow(im_out)
% 
% numImages = 2;
% imageSize = zeros(numImages,2);
% 
% for i = 1:numel(H)           
%     [xlim(i,:), ylim(i,:)] = outputLimits(H(i), [1 imageSize(i,2)], [1 imageSize(i,1)]);
% end
% xlim
% ylim


end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MAIN FUNCTION ENDED; OTHER FUNCTIONS START BELOW %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [bestFit, bestErr] = SIFTRANSAC(data, n, k, t, N, methodCol)
    % takes data as the pixel map made from SIFT files
    bestErr = Inf;
    for itr=1:k
        % maybeInliers
        maybeInliers = [];
        for i=1:n
            row = ceil(rand * size(data, 1));
            maybeInliers = [maybeInliers; data(row, :)];
        end
        % maybeModel
        model = DLT(round(maybeInliers(:,1:2)), round(maybeInliers(:,5:6)), n);
        alsoInliers = [];
        [data_diff, ~] = setdiff(data, maybeInliers, 'rows');
        for p=1:size(data_diff, 1)
            measure = data_diff(p, methodCol);
            if measure < t
                alsoInliers = [alsoInliers; data(p, :)];
            end
            if size(alsoInliers, 1) > N
                % found a good model
                allInliers = [maybeInliers; alsoInliers];
                betterModel = DLT(round(allInliers(:, 1:2)), round(allInliers(:, 5:6)), size(allInliers, 1));
                currErr = rms(round(allInliers(:, 1:2)) - round(allInliers(:, 5:6)), 'all');
                if currErr < bestErr
                    bestFit = betterModel;
                    bestErr = currErr;
                end
            end
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function k = findk(w, n, p)
    % w is an inlier ratio
    t1 = log(1-p);
    t2 = log(1-(w^n));
    k = ceil(t1/t2);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function nP = homogenousLSmodel(data, d_2, unit_vector_n, thresh)
    % we use homogenous LS model to compute the perpendicular distance 
    % from the origin to the line d; nP refers to nearby points
    dist = abs(data*unit_vector_n - d_2);
    bool = dist < thresh;
    nP = data.*bool;
    abs_d_i_2 = dist.*bool;
    nP(nP(:, 1) == 0, :) = [];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function H = DLT(X, Xp, n)
    X = [X, ones(n, 1)];
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function[pixel_map] = slideMatchesToFrames(f1_sifted, s1_sifted)
%Find best slide keypoint for every frame keypoint
%using nearest neighbor computed from euclid. distance

    length_f1 = size(f1_sifted);
    length_f1 = length_f1(1);
    length_s1 = size(s1_sifted);
    length_s1 = length_s1(1);
    
    f1_mapping = [];
    f1_scores = []; %some different ways to measure distances
    f1_angles = [];
    chi_squared = [];
    for i = 1:length_f1 %Loop through frames
        best_index = 1;
        current = f1_sifted(i,5:132);
        curr_best = s1_sifted(1,5:132);
        for j = 1:length_s1 %checking every slide vector
            if fvec_distance(current,s1_sifted(j,5:132)) < fvec_distance(current, curr_best);
                best_index = j;
                curr_best = s1_sifted(j, 5:132);
            end
        end
        f1_scores(i) = fvec_distance(current, curr_best);
        f1_mapping(i) = best_index;
        f1_angles(i) = vangle(current, curr_best);
        chi_squared(i) = chiSquared(current, curr_best);
    end

    % After generating the mapping, make an array that maps by pixels
    pixel_map = [];
  
    
    for i = 1:length_f1
        pixel_map(i,1:4) = f1_sifted(i,1:4);
        pixel_map(i,5:8) = s1_sifted(f1_mapping(i), 1:4);
        pixel_map(i,9) = f1_scores(i);
        pixel_map(i,10) = f1_angles(i);
        pixel_map(i,11) = chi_squared(i);
    end
    
    pixel_map(:,1:2) = round(pixel_map(:,1:2));
    pixel_map(:, 5:6) = round(pixel_map(:, 5:6));

    return 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Get euclidean distance between 2 sift feature vectors
function[eudist] = fvec_distance(fv1, fv2)
    fv3 = fv2-fv1;
    eudist = sqrt( sum(fv3.^2) );
    return
end

function[rad] = vangle(u,v)
    cosine = dot(u,v)/(norm(u)*norm(v));
    rad = acos(cosine);
    return
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Return the chi squared value bewtween two feature vectors
%Treating them as a pair of histograms
function[csq] = chiSquared(h1,h2)
    nums = (h1-h2).^2;
    dens = h1 + h2;
    arr = nums./dens;
    s = sum(arr);
    csq = 0.5*s;
    return 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%