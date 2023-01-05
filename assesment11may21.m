%BIL Mutisya 17132235
%%START of machine learning
clc;    % Clear the command window.
close all;  % Close all figures
clear;  % Erase all existing variables.
workspace;  % Open workspace panel
format long g;%format
format compact;%format size
fontSize = 16;%text font


%Pick image
folder = fileparts(which('Image01.jpg')); % Find folder image is in
%%folder = fileparts(which('Image01.jpg'));
% Produce a list of images
imageFiles = [dir(fullfile(folder,'*.TIF')); dir(fullfile(folder,'*.jpg'))];
for k = 1 : length(imageFiles)
    % 	fprintf('%d: %s\n', k, files(k).name);
    [~, baseFileName, extension] = fileparts(imageFiles(k).name);
    ca{k} = [baseFileName, extension];
end
% Sort the base file names alphabetically.
[ca, sortOrder] = sort(ca);
imageFiles = imageFiles(sortOrder);
button = menu('Use which gray scale demo image?', ca); % Display all image file names in a menu.
% Get the base filename.
baseFileName = imageFiles(button).name; % Assign the one on the button that they clicked on.
% Get the full filename, with path prepended.
fullFileName = fullfile(folder, baseFileName);
% fullFileName = 'C:\Users\Mark\Documents\MATLAB\work\Tests\cotton25.jpg'


% Read  image chosen
if ~exist(fullFileName, 'file')
    % Didn't find it there.  Check the search path for it.
    fullFileName = baseFileName; % No path this time.
    if ~exist(fullFileName, 'file')
        % Still didn't find it.  Alert user.
        errorMessage = sprintf('Error: %image does not exist.', fullFileName);
        uiwait(warndlg(errorMessage));
        return;
    end
end

%crop image
%for image1.jpg crop
%crop = imread(fullFileName);

% nRows=11200:12000;
% nColumns=11700:12000;
% %nColour=:
% rgbImage=crop(nRows,nColumns,:);
% %rgbImage=he(11200:12000,11700:12000,:);
% imshow(rgbImage);

%for image01.jpg do not crop crop
rgbImage= imread(fullFileName);
% Get the dimensions of the image.  numberOfColorChannels should be = 3.
[rows, columns, numberOfColorChannels] = size(rgbImage);

% Display the original color image.
subplot(2, 2, 1);
imshow(rgbImage);
title('Original Color Image', 'FontSize', fontSize, 'Interpreter', 'None');
% Enlarge figure to full screen.
set(gcf, 'Units', 'Normalized', 'Outerposition', [0, 0, 1, 1], 'Name', 'Color Channels');


%  KMEANS CLASSIFICATION
% Extract the individual red, green, and blue color channels.
redChannel = rgbImage(:, :, 1);
greenChannel = rgbImage(:, :, 2);
blueChannel = rgbImage(:, :, 3);

% Display the color channels.
subplot(2, 2, 2);
imshow(redChannel);
title('Red Channel Image', 'FontSize', fontSize, 'Interpreter', 'None');
subplot(2, 2, 3);
imshow(greenChannel);
title('Green Channel Image', 'FontSize', fontSize, 'Interpreter', 'None');
subplot(2, 2, 4);
imshow(blueChannel);
title('Blue Channel Image', 'FontSize', fontSize, 'Interpreter', 'None');

%FIRST METHOD

% Pick number of color classes
defaultValue = 5;
titleBar = 'Enter an integer value';
userPrompt = 'Enter  number of color classes to find (from 2 through 6)';
caUserInput = inputdlg(userPrompt, titleBar, 1, {num2str(defaultValue)});
if isempty(caUserInput),return,end;
% Bail out if they clicked Cancel.
% Round to nearest integer in case they entered a floating point number.
numberOfClasses = round(str2double(cell2mat(caUserInput)));
% Check for a valid integer.
if isnan(numberOfClasses) || numberOfClasses < 2 || numberOfClasses > 6
    % They didn't enter a number.
    % They clicked Cancel, or entered a character, symbols, or something else not allowed.
    numberOfClasses = defaultValue;
    message = sprintf(' integer.\nTry replacing the user.\nI will use %d and continue.', numberOfClasses);
    uiwait(warndlg(message));
end

% color segmentation by kmeans classification.

% Get the data for doing kmeans.  
% 3 columns, each with one color channel.
%  cast data it to double or else the kmeans will throw an error for uint8 data.
data = double([redChannel(:), greenChannel(:), blueChannel(:)]);
% Each row of data represents one pixel.
%  kmeans pick cluster each pixel belongs to.
indexes = kmeans(data, numberOfClasses);
%convert what class index the pixel is in into images for each class index.
class1 = reshape(indexes == 1, rows, columns);
class2 = reshape(indexes == 2, rows, columns);
class3 = reshape(indexes == 3, rows, columns);
class4 = reshape(indexes == 4, rows, columns);
class5 = reshape(indexes == 5, rows, columns);
class6 = reshape(indexes == 6, rows, columns);

%  put them into a 3-D array for easier display once in loop.
allClasses = cat(3, class1, class2, class3, class4, class5, class6);
allClasses = allClasses(:, :, 1:numberOfClasses); % Crop off just what is need.
%  display our classification images.

% Plot the 3-D color gamut
colorcloud(rgbImage);
title('RGB Color Cloud (3-D Color Gamut)', 'FontSize', fontSize, 'Interpreter', 'None');
% Enlarge figure
set(gcf, 'Units', 'Normalized', 'Outerposition', [0.05, 0.05, 0.9, 0.9], 'Name', 'Color classes');

% Create indexed image for comparison;
[indexedImage, customColorMap]  = rgb2ind(rgbImage, numberOfClasses);
figure; % Bring up new figure.
% Display the color channels again on the new figure.
subplot(3, numberOfClasses, 1);
imshow(rgbImage);
title('RGB Color Image', 'FontSize', fontSize, 'Interpreter', 'None');
% subplot(3, numberOfClasses, 2);
h3 = subplot(3, numberOfClasses, 3);
imshow(indexedImage, []);
colormap(h3, customColorMap);
colorbar;
title('Indexed (quantized) Image using rgb2ind()', 'FontSize', fontSize, 'Interpreter', 'None');
% Enlarge figure to near full screen.
set(gcf, 'Units', 'Normalized', 'Outerposition', [0.1, 0.1, 0.8, 0.8], 'Name', 'Color classes');


% Display the classes, both binary and masking the original.
%  indexes image in order to display each class in a unique color.
indexedImageK = zeros(size(indexedImage), 'uint8'); % Initialize another indexed image.
for c = 1 : numberOfClasses
    % Display binary image of  pixels using  class ID number.
    subplot(3, numberOfClasses, c + numberOfClasses);
    thisClass = allClasses(:, :, c);
    imshow(thisClass);
    caption = sprintf('Image of\nClass %d Indexes', c);%insert caption
    title(caption, 'FontSize', fontSize);%title of mage
    
    % Mask the image using bsxfun() function
    maskedRgbImage = bsxfun(@times, rgbImage, cast(thisClass, 'like', rgbImage));
    % Display masked image.
    subplot(3, numberOfClasses, c + 2 * numberOfClasses);
    imshow(maskedRgbImage);
    caption = sprintf('Class %d Image\nMasking Original', c);
    title(caption, 'FontSize', fontSize);
    
    % Make indexed image
    indexedImageK(thisClass) = c;
end

% Display the image, indexed by kmeans, in pseudocolor.
h5 = subplot(3, numberOfClasses, 5);
imshow(indexedImageK, customColorMap);
% colormap(h5, customColorMap); % Use the same colormap as rgb2ind() used.
colorbar;
title('Indexed (quantized) Image using kmeans()', 'FontSize', fontSize, 'Interpreter', 'None');
%Note: class numbers assigned by rgb2ind and kmeans may be different, so the color at each pixel may differ'

%Features
% 1)counting nucleas
%%Segment the Nuclei count
%check the color channels for the average nucleau and then set the variable to count
bluePixels =  maskedRgbImage(:,:,1)<=50 & maskedRgbImage(:,:,2)<=30 & maskedRgbImage(:,:,3)>=50;
nucleousnumber = sum(maskedRgbImage(:));

%%



%SECOND METHOD
%Color-Based Segmentation Using K-Means Clustering
%https://uk.mathworks.com/help/images/color-based-segmentation-using-k-means-clustering.html

% he = imread('Image1.jpg');
% [rows, columns, numberOfColorChannels] = size(he);
% nRows=11200:12000;
% nColumns=11700:12000;
% rgbImage=he(nRows,nColumns,:);
% figure,imshow(rgbImage);

%rgbImage = imread('Image01.jpg');
% Convert Image from RGB Color Space to L*a*b* Color Space
lab_he = rgb2lab(rgbImage);
%imshow(rgbImage);
%Classify the Colors in 'a*b*' Space Using K-Means Clustering
ab = lab_he(:,:,2:3);
ab = im2single(ab);
nColors = 3;
% repeat the clustering 3 times to avoid local minima
pixel_labels = imsegkmeans(ab,nColors,'NumAttempts',3);
figure,imshow(pixel_labels,[])
title('Image Labeled by Cluster Index');
% Create Images that Segment the H&E Image by Color
mask1 = pixel_labels==1;
cluster1 = rgbImage.* uint8(mask1);
imshow(cluster1)
title('Objects in Cluster 1');

mask2 = pixel_labels==2;
cluster2 = rgbImage.* uint8(mask2);
figure,imshow(cluster2)
title('Objects in Cluster 2');

figure,mask3 = pixel_labels==3;
cluster3 = rgbImage.* uint8(mask3);
imshow(cluster3)
title('Objects in Cluster 3');
%cluster 3 ia the clearest veiw of nucleaus therefore its  mask is used in segmnrting nuclues
%it proved ineffecient so segmentation of nucleus needed to be performed

%Segment the Nuclei
%first mask is the most effecient
L = lab_he(:,:,1);
L_blue = L .* double(mask1);
L_blue = rescale(L_blue);
idx_light_blue = imbinarize(nonzeros(L_blue));

blue_idx = find(mask1);
mask_dark_blue = mask1;
mask_dark_blue(blue_idx(idx_light_blue)) = 0;

blue_nuclei = rgbImage.* uint8(mask_dark_blue);
figure,imshow(blue_nuclei)
title('Blue Nuclei');
%can use region props for finding nucles
%bwareaopena - to remove non nucleus

%--------FEATURE EXTRACTION--------
% 1)counting nucleas
%%Segment the Nuclei count
%check the color channels for the average nucleau and then set the variable to count
bluePixels2 =  blue_nuclei(:,:,1)<=90 & blue_nuclei(:,:,2)<=40 & blue_nuclei(:,:,3)>=50;
nucleousnumber2 = sum(bluePixels(:));

% 2) Histogram
%%Adaptive Histogram Equalization
%Image=imread('Image01.jpg');
%grayImage = rgb2gray(fullFileName);
%I= rgb2gray(Image);
I= rgb2gray(rgbImage);
%I = rgb2gray(fullFileName);

%contrast-adjusted image with its histogram.
J = histeq(I);%Enhance contrast using histogram equalization
K=adapthisteq(I);%Contrast-limited adaptive histogram equalization
f2=figure(2);
set(f2,'Position',[ 0 0 800 400]);
subplot (231) ,imshow (I), title ('original ');
subplot (232) ,imshow (J) , title ('contrast ');
subplot (233) ,imshow (K), title (' locally adaptive histogram equalization ');
subplot (234) , imhist(I,64);title ('original histogram ');
subplot (235) ,imhist(J,64);title ('contast histogram ');
subplot (236) ,imhist(K,64);title ('adaptive histogram ');



%%

%3)local feature
% measure cells and nucleaus size

%https://uk.mathworks.com/matlabcentral/answers/86610-pointing-to-a-certain-area-and-calculate-the-average
% freehand draw an irregular shape over a gray scale image.

%for image1.jpg crop
% he = imread('Image1.jpg');
% %[rows, columns, numberOfColorChannels] = size(he);
% %for i=11200:11200:row
%  %    for j=11700:12000:col
% %hes=he(11200:12000,11700:12000
%nRows=11200:12000;
% nColumns=11700:12000;
% %nColour=:
% rgbImage=he(nRows,nColumns,:);
% grayImage = rgb2gray(rgbImage);

%for image01.jpg  crop
nRows=402:1030;
nColumns=178:819;
 crop=rgbImage(nRows,nColumns,:);


grayImage = rgb2gray(crop);

figure,imshow(grayImage, []);
axis on;
title('Original Grayscale Image', 'FontSize', fontSize);
set(gcf, 'Position', get(0,'Screensize')); % Maximize figure.
message = sprintf('Draw the shape on Original Grayscale Image \n Left click and hold to begin drawing.\n stop holdind button to finish');
uiwait(msgbox(message));
hFH = imfreehand();
% Create a binary image ("mask") from the ROI object.
binaryImage = hFH.createMask();
xy = hFH.getPosition;

%  make it smaller to view more images.
subplot(2, 3, 1);
imshow(grayImage, []);
axis on;
drawnow;
title('Original Grayscale Image', 'FontSize', fontSize);

% Display the freehand mask.
subplot(2, 3, 2);
imshow(binaryImage);
axis on;
title('Binary mask of the region', 'FontSize', fontSize);

% Label the binary image and calculate the centroid and center of mass.
labeledImage = bwlabel(binaryImage);
measurements = regionprops(binaryImage, grayImage, ...
    'area', 'Centroid', 'WeightedCentroid', 'Perimeter');
area = measurements.Area
centroid = measurements.Centroid
centerOfMass = measurements.WeightedCentroid
perimeter = measurements.Perimeter

% Calculate the area, in pixels, that they drew.
numberOfPixels1 = sum(binaryImage(:))
% Another way to calculate it that takes fractional pixels into account.
numberOfPixels2 = bwarea(binaryImage)

% Get coordinates of the boundary of the freehand drawn region.
structBoundaries = bwboundaries(binaryImage);
xy=structBoundaries{1}; % Get n by 2 array of x,y coordinates.
x = xy(:, 2); % Columns.
y = xy(:, 1); % Rows.
subplot(2, 3, 1); % Plot over original image.
hold on; % Don't blow away the image.
plot(x, y, 'LineWidth', 2);
drawnow; % Force it to draw immediately.

% Burn line into image by setting it to 255 .This applies where the mask is true.
burnedImage = grayImage;
burnedImage(binaryImage) = 255;
% Display the image with the mask "burned in."
subplot(2, 3, 3);
imshow(burnedImage);
axis on;
caption = sprintf('New image with\nmask burned into image');
title(caption, 'FontSize', fontSize);

% Mask the image and display it.
%This will keep  the parts of the image that's inside the mask, zero outside mask.
blackMaskedImage = grayImage;
blackMaskedImage(~binaryImage) = 0;
subplot(2, 3, 4);
imshow(blackMaskedImage);
axis on;
title('Masked Outside Region', 'FontSize', fontSize);
% Calculate the mean
meanGL = mean(blackMaskedImage(binaryImage));
sdGL = std(double(blackMaskedImage(binaryImage)));

% Put up crosses at the centriod and center of mass
hold on;
plot(centroid(1), centroid(2), 'r+', 'MarkerSize', 30, 'LineWidth', 2);
plot(centerOfMass(1), centerOfMass(2), 'g+', 'MarkerSize', 20, 'LineWidth', 2);

% Now do the same but blacken inside the region.
insideMasked = grayImage;
insideMasked(binaryImage) = 0;
subplot(2, 3, 5);
imshow(insideMasked);
axis on;
title('Masked Inside Region', 'FontSize', fontSize);

% Now crop the image.
leftColumn = min(x);
rightColumn = max(x);
topLine = min(y);
bottomLine = max(y);
width = rightColumn - leftColumn + 1;
height = bottomLine - topLine + 1;
croppedImage = imcrop(blackMaskedImage, [leftColumn, topLine, width, height]);
% Display cropped image.
subplot(2, 3, 6);
imshow(croppedImage);
axis on;
title('Cropped Image', 'FontSize', fontSize);

% Put up crosses at the centriod and center of mass
hold on;
plot(centroid(1)-leftColumn, centroid(2)-topLine, 'r+', 'MarkerSize', 30, 'LineWidth', 2);
plot(centerOfMass(1)-leftColumn, centerOfMass(2)-topLine, 'g+', 'MarkerSize', 20, 'LineWidth', 2);

% Report results.
message = sprintf('Mean value within drawn area = %.3f\nStandard deviation within drawn area = %.3f\nNumber of pixels = %d\nArea in pixels = %.2f\nperimeter = %.2f\nCentroid at (x,y) = (%.1f, %.1f)\nCenter of Mass at (x,y) = (%.1f, %.1f)\nRed crosshairs at centroid.\nGreen crosshairs at center of mass.', ...
    meanGL, sdGL, numberOfPixels1, numberOfPixels2, perimeter, ...
    centroid(1), centroid(2), centerOfMass(1), centerOfMass(2));
msgbox(message);





