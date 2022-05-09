%%% Marium Yousuf, Kayla Bennett
%%% CSC 577
%%% Assignment 11
function hw11D1()
% this function incorporates all the programming for hw11
% for CSC 577 - Intro to Computer Vision
close all
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PART D1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
im1 = imread("FM-inside-one.jpg");
% imshow(im1);
% datacursormode on
% img_coords = ginput(20);
% writematrix(img_coords, 'FM_1.txt', 'Delimiter', 'tab');
% datacursormode off

im2 = imread("FM-inside-two.jpg");
% imshow(im2);
% datacursormode on
% img_coords = ginput(20);
% writematrix(img_coords, 'FM_2.txt', 'Delimiter', 'tab');
% datacursormode off

FM_1_kp = readmatrix("mouse-click_coords/FM_1.txt");
% row_FM_1_kp = cast(FM_1_kp(:, 2), 'uint32');
% col_FM_1_kp = cast(FM_1_kp(:, 1), 'uint32');
% imshow(im1);
% hold on
% plot(col_FM_1_kp, row_FM_1_kp, 'ws', 'MarkerSize', 5, 'LineWidth', 2);
% hold off
% 
FM_2_kp = readmatrix("mouse-click_coords/FM_2.txt");
% row_FM_2_kp = cast(FM_2_kp(:, 2), 'uint32');
% col_FM_2_kp = cast(FM_2_kp(:, 1), 'uint32');
% imshow(im2);
% hold on
% plot(col_FM_2_kp, row_FM_2_kp, 'ws', 'MarkerSize', 5, 'LineWidth', 2);
% hold off

rand_L = randperm(size(FM_2_kp, 1)); 
% randomly assign training and testing indices
train_idx = rand_L(1:0.6*size(FM_2_kp, 1));
test_idx = setdiff(1:size(FM_2_kp, 1), train_idx);

% assign training points and testing points
train_im1 = FM_1_kp(train_idx, :);
train_im1 = [train_im1, ones(size(train_im1, 1), 1)];
train_im2 = FM_2_kp(train_idx, :);
train_im2 = [train_im2, ones(size(train_im2, 1), 1)];
test_im1 = FM_1_kp(test_idx, :);
test_im1 = [test_im1, ones(size(test_im1, 1), 1)];
test_im2 = FM_2_kp(test_idx, :);
test_im2 = [test_im2, ones(size(test_im2, 1), 1)];

F = computeFundamentalMx(train_im1, train_im2);
error = zeros([1, size(test_im2, 1)]);
for i=1:size(test_im2, 1)
    val = test_im1(i,:) * F * (test_im2(i,:)');
    error(i) = val;
end
err = rms(error);
new_img1 = epipolarLines(im1, FM_1_kp, F);
new_img2 = epipolarLines(im2, FM_2_kp, F);
row_FM_1_kp = cast(FM_1_kp(:, 2), 'uint32');
col_FM_1_kp = cast(FM_1_kp(:, 1), 'uint32');
row_FM_2_kp = cast(FM_2_kp(:, 2), 'uint32');
col_FM_2_kp = cast(FM_2_kp(:, 1), 'uint32');
imshow(new_img1);
hold on
plot(col_FM_1_kp, row_FM_1_kp, 'ws', 'MarkerSize', 5, 'LineWidth', 2);
hold off

% imshow(new_img2);
% hold on
% plot(col_FM_2_kp, row_FM_2_kp, 'ws', 'MarkerSize', 5, 'LineWidth', 2);
% hold off

end

function F = computeFundamentalMx(p1, p2)
    numPts = size(p1, 1);
    mx = [];
    for i=1:numPts
        mat = [p1(i, 1).*p2(i, :), p1(i, 2).*p2(i, :), p1(i, 3).*p2(i, :)];
        mx = [mx; mat];
    end
    U_temp = mx'*mx;
    [V, D] = eig(U_temp);
    D = diag(D);
    [~, m] = min(D);
    H_vec = V(:, m);
    F = reshape(H_vec, 3, 3)';
end

function new_img = epipolarLines(img, points, F)
    numPts = size(points, 1);
    new_img = img;
    for i=1:numPts(1)
        x = [points(i,:), 1];
        line = F*x';
        new_img = draw_line(new_img, line, 1, 255, 255, 0);
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% This function draws the line represented by the second argument which is
% vector [a,b,d] that describes a line by a*i+b*j+d = 0. It returns a copy of the
% image (first argument) with the line drawn into it. The last 4 arguments are
% the width of the line, and its color in R, G, and B. 
%
% This code requires the function draw_segment() to be in your Matlab path. 
%
% Previously, the name draw_line() was given to a different function. That
% function is now called "draw_segment". 
%
function new_image = draw_line(image, coef, width, red, green, blue)
   % Clip the line against the box, and then draw it.

   new_image = image; 

   m = size(image,1);
   n = size(image,2);

   num_found = 0;
   p = [];

   a = coef(1); b = coef(2); d = coef(3);

   if (abs(b) > 1e-5) 
       j_test = - (d + a) / b;

       % if ((j_test >= 1) & (j_test <= n))
       if ((j_test > -width) & (j_test < n+width))
          num_found = num_found + 1;
          p(num_found, : ) = [ 1  j_test ];
       end 

       j_test = - (d + a*m) / b;

       % if ((j_test >= 1) & (j_test <= n)) 
       if ((j_test > -width) & (j_test < n+width)) 
          num_found = num_found + 1;
          p(num_found, : ) = [ m  j_test ];
       end  
   end  

   if (abs(a) > 1e-5) 
       if (num_found < 2) 
           i_test = - (d + b) / a;

           if ((i_test > -width) & (i_test < m+width))
           % if ((i_test >= 1) & (i_test <= m))
              num_found = num_found + 1;
              p(num_found, : ) = [ i_test  1 ];
           end 
       end 
           
       if (num_found < 2) 
           i_test = - (d + n*b) / a;

           if ((i_test > -width) & (i_test < m+width)) 
           % if ((i_test >= 1) & (i_test <= m)) 
              num_found = num_found + 1;
              p(num_found, : ) = [ i_test  n ];
           end 
       end 
   end 

   if (num_found == 2) 
       new_image = draw_segment(new_image, p(1,:), p(2,:), width, red, green, blue);
   end 
end
       
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

