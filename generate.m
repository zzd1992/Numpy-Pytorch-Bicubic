% size of original image should be multiples of 12

x = imread('lenna.png');
x = im2double(x);

y = imresize(x, 1.0/2);
imwrite(y, 'lenna_down_2.png');
y = imresize(x, 1.0/3);
imwrite(y, 'lenna_down_3.png');
y = imresize(x, 1.0/4);
imwrite(y, 'lenna_down_4.png');

y = imresize(x, 2);
imwrite(y, 'lenna_up_2.png');
y = imresize(x, 3);
imwrite(y, 'lenna_up_3.png');
y = imresize(x, 4);
imwrite(y, 'lenna_up_4.png');