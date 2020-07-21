clc
clear all

%Create image datastore
imds = imageDatastore(fullfile('Flowers'),'IncludeSubfolders',true,'FileExtensions','.jpg','LabelSource','foldernames');

%Count number of images per label and save the number of classes
labelCount = countEachLabel(imds);
numClasses = height(labelCount);

%Create training and validation sets
[imdsTraining, imdsValidation] = splitEachLabel(imds, 0.7);
inputSize = [224,224,3];

%Use image data augmentation to handle the resizing the original images are 256-by-256. 
%The input layer of the CNNs used in this example expects them to be 224-by-224.
augimdsTraining = augmentedImageDatastore(inputSize(1:2),imdsTraining);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

%We build a simple CNN.
%specify its training options, train it, and evaluate it.
%Define Layers
layers = [
imageInputLayer([224 224 3])
convolution2dLayer(3,16,'Padding',1)
batchNormalizationLayer
reluLayer
maxPooling2dLayer(2,'Stride',2)
convolution2dLayer(3,32,'Padding',1)
batchNormalizationLayer
reluLayer
maxPooling2dLayer(2,'Stride',2)
convolution2dLayer(3,64,'Padding',1)
batchNormalizationLayer
reluLayer
fullyConnectedLayer(12)
softmaxLayer
classificationLayer];

%Specify Training Options
options = trainingOptions('sgdm',...
'MaxEpochs',30, ...
'ValidationData',augimdsValidation,...
'ValidationFrequency',50,...
'InitialLearnRate', 0.0003,...
'Verbose',false,...
'Plots','training-progress');

%Train network
%Classify and Compute Accuracy
baselineCNN = trainNetwork(augimdsTraining,layers,options);
predictedLabels = classify(baselineCNN,augimdsValidation);
valLabels = imdsValidation.Labels;
baselineCNNAccuracy = sum(predictedLabels == valLabels)/numel(valLabels);
