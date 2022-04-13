%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         adversarialAttack_V1
% Description:  Apply resnet 18 to cinic 10
% Authors:      Luong-Ha Nguyen & James-A. Goulet
% Created:      May 18, 2021
% Updated:      May 18, 2021
% Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
% Copyright (c) 2021 Luong-Ha nguyen & James-A. Goulet 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear 
% clc
close all
set(0,'DefaultAxesFontName','Helvetica')
set(0,'defaultLineLineWidth',2)
set(0,'DefaultAxesFontSize',14)
set(0,'defaulttextfontsize',14)
format shortE
rng(1223)
% gpuDevice(1);
%% Data
Nsamples  = 20;
net.testing = true;
modelName = 'resnet18_V1';
dataName  = 'cifar10';
ltrain    = load('data/cifar10train.mat');
trainImdb = ltrain.trainImdb;
ltest     = load('data/cifar10test.mat');
testImdb  = ltest.testImdb;
testImdb.img         = testImdb.img(:, :, :, 1 : Nsamples);
testImdb.digitlabels = testImdb.digitlabels(1 : Nsamples);
testImdb.classObs    = trainImdb.classObs;
testImdb.classIdx    = trainImdb.classIdx;
testImdb.numImages   = Nsamples;
imgSize   = [32 32 3];
% imgStat   = [0.485, 0.456, 0.406; % mean
%              0.229, 0.224, 0.225];% std
imgStat   = [0.485, 0.456, 0.406; % mean
             1, 1, 1];% std   
%% Neural Network properties
net.task           = 'classification';
net.modelName      = modelName;
net.dataName       = dataName;
net.cd             = cd;
% Save net after N epoch
net.maxEpoch       = 50; 
net.savedEpoch     = 10;
% GPU 1: yes; 0: no
net.gpu            = true;
net.cuda           = true;
net.numDevices     = 1;
% Data type object half or double precision
net.dtype          = 'single';
% Number of input covariates
net.imgSize        = imgSize;
net.resizeMode     = false;
net.resizedSize    = imgSize;
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
% Number of nodes for each layer| 1: FC; 2:conv; 3: pooling
net.imgSize        = imgSize;
net                = indices.layerEncoder(net);
[net.layer, net.filter, net.kernelSize, net.padding, net.paddingType, net.stride, net.nodes, net.actFunIdx, net.xsc] = resnet.imagenet18_10classes(imgSize, net.ny, 2);
% Observations standard deviation
net.learnSv        = 0;
net.sv             = 0*ones(1,1, net.dtype);      
net.svDecayFactor  = 0.975;
net.svmin          = 0.2;
net.Sx             = cast(0.1.^2, net.dtype);
% Runing average
net.normMomentum   = 0.9;

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
trainedModelDir = ['results/resnet18_V3_E50_cifar10']; 
initModelDir    = [];
initEpoch = 0;

%% Run
task.adversarialAttack(net, testImdb, trainedModelDir, initEpoch)

