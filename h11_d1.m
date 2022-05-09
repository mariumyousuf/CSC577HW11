%% File input for D1
im1 = imread('IMG_7345.jpg');
im2 = imread('IMG_7346.jpg');

%Reading in points from each image. 
%Order: Left to right, front to back
%XY notation
%% Read them boys
datacursormode ON
figure();
imshow(im1);
%%
figure();
imshow(im2);

%% Put points in from indices
pts1 = [
    1369 515
    1991 732
    1949 971
    1800 1176
    1751 1294
    1229 1629
    1390 1492
    1765 1420
    1284 2125
    1152 2186
    2264 2333
    1906 2660
    2526 2650
    1840 2867
    2862 2673
    2103 2940
    2077 3016
    2269 3496
    2166 3694
    2440 3262
]

pts2 = [
    537 536
    1118 738
    1120 977
    1027 1179
    959 1294
    329 1661
    444 1520
    1030 1416
    429 2150
    272 2207
    1580 2283
    1227 2625
    1758 2570
    1203 2834
    1994 2575
    1203 2834
    1365 2883
    1512 3405
    1474 3600
    2175 3187
]

%% Write them to a file
pts_all = [pts1, pts2];
writematrix(pts_all, "points_xy.csv");