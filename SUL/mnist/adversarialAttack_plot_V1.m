%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         adversarialAttack_plot_V1
% Description:  plot adversarial-attack images on Mnist
% Authors:      Luong-Ha Nguyen & James-A. Goulet
% Created:      May 24, 2021
% Updated:      May 24, 2021
% Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
% Copyright (c) 2021 Luong-Ha nguyen & James-A. Goulet 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
% clc
% close all
set(0,'DefaultAxesFontName','Helvetica')
set(0,'defaultLineLineWidth',2)
set(0,'DefaultAxesFontSize',14)
set(0,'defaulttextfontsize',14)
format shortE
% gpuDevice(1);
rng(1)
%% Data
Nsamples  = 20;
modelName = 'dropconnect_V2';
dataName  = 'mnist';
train     = load('data/mnistTrain.mat');
trainImdb = train.trainImdb;
test      = load('data/mnistTest.mat');
testImdb  = test.testImdb;
[labels, idx]  = unique(testImdb.digitlabels);
testImdb.img         = testImdb.img(:, :, :, idx);
testImdb.digitlabels = testImdb.digitlabels(idx);
testImdb.classObs    = trainImdb.classObs;
testImdb.classIdx    = trainImdb.classIdx;
testImdb.numImages   = length(idx);
imgSize   = [28 28 1];
imgStat   = [1.3093e-01; % mean
             1];% std

%% Neural Network properties
net.task           = 'classification';
net.modelName      = modelName;
net.dataName       = dataName;
net.cd             = cd;
% Save net after N epoch
net.maxEpoch       = 100; 
net.savedEpoch     = round(net.maxEpoch/2);
% GPU 1: yes; 0: no
net.gpu            = true;
net.cuda           = true;
net.numDevices     = 1;
% Data type object half or double precision
net.dtype          = 'single';
% Number of input covariates
net.imgSize        = imgSize;
net.resizeMode     = false;
net.resizedSize    = nan;
net.nx             = prod(net.imgSize); 
net.imgStat        = imgStat;
% Number of output responses
ny                 = trainImdb.numOutputs; 
net.ny             = ny;   
net.nl             = ny;
net.nv2            = ny; 
net.nye            = trainImdb.numEnOutputs; % Num. of hidden states
% Number of classes
net.numClasses     = trainImdb.numClasses;
% Batch size 
net.batchSize      = 10; 
net.repBatchSize   = 1;
% Layer| 1: FC; 2:conv; 3: max pooling; 
% 4: avg pooling; 5: layerNorm; 6: batchNorm 
% Activation: 1:tanh; 2:sigm; 3:cdf; 4:relu; 5:softplus
net.imgSize        = imgSize;
net.layer          = [2       2         3       2       3      1    1];
net.filter         = [1       32        32      64      64     1    1];
net.kernelSize     = [4       3         5       3       1      1    1];
net.padding        = [1       0         0       0       0      0    0];
net.stride         = [1       2         1       2       0      0    0];
net.nodes          = [net.nx  0         0       0       0      150  net.ny]; 
net.actFunIdx      = [0       4         0       4       0      4    0];
net.paddingType    = [1       0         0       0       0      0    0];
% Observations standard deviation
net.learnSv        = 0; % Online noise learning
net.sv             = 0.0*ones(1, 1, net.dtype, 'gpuArray');  
net.svDecayFactor  = 0.975;
net.svmin          = 0.01;
net.Sx             = cast(0.03.^2, net.dtype);
% Parameter initialization
net.initParamType  = 'He';
net.gainSw         = ones(1, length(net.layer) - 1);
net.gainMw         = ones(1, length(net.layer) - 1);
net.gainSb         = ones(1, length(net.layer) - 1);
net.gainMb         = ones(1, length(net.layer) - 1);

% Last layer update
net.isUdidx         = true;
net.wxupdate        = true;
net.collectDev      = false;
net.convariateEstm  = true;
net.errorRateEval   = true;

%% Pretrained model directory
trainedModelDir = ['results/dropconnect_B16_He__V1_E50_mnist']; 
initEpoch       = 0;
%% Run
task.adversarialAttack4plot(net, testImdb, trainedModelDir, initEpoch)
