%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         tagiOptNstepsQ_V2
% Description:  nsteps Q for hopper
% Authors:      Luong-Ha Nguyen & James-A. Goulet
% Created:      February 28, 2021
% Updated:      February 28, 2021
% Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
% Copyright (c) 2021 Luong-Ha nguyen & James-A. Goulet 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
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
modelName        = 'tagiOptNstepsQ_V1';
dataName         = 'halfcheetah';
env              = halfCheetahEnv();
% RL hyperparameters 
rl.numEpisodes   = 20000;
rl.imgSize       = [1 1 1];
rl.gamma         = 0.99;
rl.lambda        = 0.9; 
rl.maxNumSteps   = 2E6;
rl.numStepsEps   = 5000;
rl.maxMemory     = 2E5;
rl.actionSpace   = 1;
rl.numActions    = 1;
rl.numActions    = double(env.open_env.action_space.shape{1});
rl.actionLow     = single(env.open_env.action_space.low)';
rl.actionHigh    = single(env.open_env.action_space.high)';
rl.numStates     = double(env.open_env.observation_space.shape{1});
rl.obsLow        = -10*ones(rl.numStates , 1);
rl.obsHigh       = 10*ones(rl.numStates , 1);
rl.stateSpace    = prod(rl.imgSize);
rl.rewardLow     = -10;
rl.rewardHigh    = 10;
rl.rewardSpace   = 1;
rl.targetUpdate  = 1; % Update the target network after N times
rl.initialMemory = 2048;
rl.stepUpdate    = 1024;
rl.updateBatch   = 16;
rl.noptepochs    = 1;
netA.rl          = rl;
netQ.rl          = rl;
netA.seed        = seed;
netQ.seed        = seed;

%% Q-value net
netQ.task           = 'regression';
netQ.modelName      = modelName;
netQ.dataName       = dataName;
netQ.cd             = cd;
netQ.savedEpisode   = round(rl.numEpisodes/2);
netQ.savedUpdate    = 20000; 
netQ.logInterval    = 100;
% GPU 1: yes; 0: no
netQ.gpu            = true;
netQ.cuda           = true;
% Data type object half or double precision
netQ.dtype          = 'single';
% Number of input covariates
netQ.nx             = netQ.rl.numStates+netQ.rl.numActions; 
% Number of output responses
netQ.ny             = 1;   
% Batch size 
netQ.batchSize      = 16; 
netQ.repBatchSize   = 1 ;
% Number of nodes for each layer| 1: FC; 2:conv; 3: pooling
netQ.imgSize        = netQ.rl.imgSize;
netQ.layer          = [1        1   1   1   1];
netQ.nodes          = [netQ.nx  128 128 128 netQ.ny];  
netQ.actFunIdx      = [0        1   4   4   0];
netQ.actBound       = [1        1   1   1   1];
% Observations standard deviation
netQ.learnSv        = 0;
netQ.sv             = 2*ones(1,1);   
netQ.svDecayFactor  = 0.9999;
netQ.svmin          = 0.3;
% Parameter initialization | {Xavier, He}
netQ.initParamType  = 'He';
netQ.gainSw          = [1   1  1   1];
netQ.gainMw          = [1   1  1   1];
netQ.gainSb          = [1   1  1   1];
netQ.gainMb          = [1   1  1   1];
% Update type
netQ.lastLayerUpdate = true; 
netQ.collectDev      = true;
netQ.convariateEstm  = true;

%% Action net
netA.task           = 'regression';
netA.modelName      = modelName;
netA.dataName       = dataName;
netA.cd             = cd;
netA.savedEpisode   = round(rl.numEpisodes/2);
% GPU 1: yes; 0: no
netA.gpu            = netQ.gpu;
netA.cuda           = netQ.cuda;
% Data type object half or double precision
netA.dtype          = 'single';
% Number of input covariates
netA.nx             = netA.rl.numStates; 
% Number of output responses
netA.ny             = netA.rl.numActions;   
% Batch size 
netA.batchSize      = netQ.batchSize ; 
netA.repBatchSize   = netQ.repBatchSize ;
% Number of nodes for each layer| 1: FC; 2:conv; 3: pooling
netA.imgSize        = netA.rl.imgSize;
netA.layer          = [1        1   1   1];
netA.nodes          = [netA.nx  128 128 netA.ny];  
netA.actFunIdx      = [0        4   4   1];
netA.actBound       = [1        1   1   1];
% Observations standard deviation
netA.learnSv        = 0;
netA.sv             = 0.2*ones(1,1);    
netA.svDecayFactor  = 0.9999;
netA.svmin          = 0.3;
% Parameter initialization | {Xavier, He}
netA.initParamType  = 'He';
netA.gainSw          = [1 1 1];
netA.gainMw          = [1 1 1];
netA.gainSb          = [1 1 1];
netA.gainMb          = [1 1 1];
% Update type
netA.lastLayerUpdate = true; 
netA.collectDev      = false;
netA.convariateEstm  = false;
netA.isUdidx         = false;

%% Load model from results's directory
trainedModelDir = [];
% trainedModelDir = ['results/tagiOptNstepsQ_pure_V7_E726_hopper']; 
startEpisode    = 0;

%% Run
task.tagiOptNstepQ(netQ, netA, env, trainedModelDir, startEpisode);


