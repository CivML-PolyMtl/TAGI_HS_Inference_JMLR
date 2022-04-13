%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         dropconetect_V3
% Description:  Apply 3 conv. layer to cifar 10
% Authors:      Luong-Ha Nguyen & James-A. Goulet
% Created:      January 13, 2021
% Updated:      August 19, 2021
% Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
% Copyright (c) 2021 Luong-Ha nguyen & James-A. Goulet. All rights reserved. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear 
% clc
close all
set(0,'DefaultAxesFontName','Helvetica')
set(0,'defaultLineLineWidth',2)
set(0,'DefaultAxesFontSize',14)
set(0,'defaulttextfontsize',14)
format shortE
seed = 0;
rng(seed)
%% Data
net.testing = true;
modelName = 'dropconetectE50_V1';
dataName  = 'cifar10';
ltrain    = load('data/cifar10train.mat');
trainImdb = ltrain.trainImdb;
trainImdb.classObs = single(trainImdb.classObs);
trainImdb.classIdx = single(trainImdb.classIdx);
ltest     = load('data/cifar10test.mat');
testImdb  = ltest.testImdb;
imgSize   = [32 32 3];
imgStat   = [0.485, 0.456, 0.406; % mean
             1, 1, 1];% std
net.seed   = seed;         
%% Neural Network properties
net.task           = 'classification';
net.modelName      = modelName;
net.dataName       = dataName;
net.cd             = cd;
% Save net after N epoch
net.maxEpoch       = 10; 
net.savedEpoch     = round(net.maxEpoch/2);
% GPU 1: yes; 0: no
net.gpu            = true;
net.cuda           = true;% run cuda compiler version?
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
% Number of nodes for each layer| 1: FC; 2:conv; 3: pooling; 5: LayerNorm
% 6: batchNorm
% Activation: 1: tanh; 2: sigmoid; 3: cdf; 4: ReLU; 5: softplus.
net.imgSize        = imgSize;
net                = indices.layerEncoder(net);
net.layer          = [2       2   6   4       2   6   4       2   6   4    1    1];
net.filter         = [3       32  32  32      32  32  32      64  64  64   1    1];
net.kernelSize     = [5       1   3   5       1   3   5       1   3   1    1    1];
net.padding        = [2       0   1   2       0   1   2       0   2   0    0    0];
net.stride         = [1       0   2   1       0   2   1       0   1   0    0    0];
net.nodes          = [net.nx  0   0   0       0   0   0       0   0   0    64   net.ny];  
net.actFunIdx      = [0       4   0   0       4   0   0       4   0   0    4    0];
net.paddingType    = [1       0   2   1       0   2   1       0   2   0    0    0];
net.actBound       = ones(size(net.layer));

% Observations standard deviation
net.sv             = 4*ones(1,1);      
net.svDecayFactor  = 0.975;
net.svmin          = 0.3;
net.lastLayerUpdate = true; 

% Misc
net.isUdidx        = true;
net.errorRateEval  = true;

% Factor for initializing weights & bias | {Xavier, He}
net.initParamType  = 'He';

% Data augmentation
net                = indices.daTypeEncoder(net);
net.da.enable      = false;
net.obsShow        = 1000;
%% Pretrained model directory
% trainedModelDir = ['results/dropconetectE50_BN_V1_E50_cifar10']; 
trainedModelDir = [];
initModelDir    = [];
initEpoch       = 0;

%% Run
task.runClassification(net, trainImdb, testImdb, trainedModelDir, initModelDir, initEpoch);

