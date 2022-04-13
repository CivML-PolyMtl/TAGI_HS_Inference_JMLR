%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         tagiOptNstepsQ_V1
% Description:  nsteps Q for  InvertedPendulumEnv
% Authors:      Luong-Ha Nguyen & James-A. Goulet
% Created:      January 12, 2021
% Updated:      August 19, 2021
% Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
% Copyright (c) 2021 Luong-Ha nguyen & James-A. Goulet. All right reserved. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear 
% clc
close all
set(0,'DefaultAxesFontName','Helvetica')
set(0,'defaultLineLineWidth',2)
set(0,'DefaultAxesFontSize',14)
set(0,'defaulttextfontsize',14)
format shortE
seed = 1;
rng(seed)
%% Data
modelName        = 'nstepQ_contAction_1';
dataName         = 'cartPole';
env              = InvertedPendulumEnv();
% RL hyperparameters 
rl.numEpisodes   = 20000;
rl.imgSize       = [1 1 1];
rl.gamma         = 0.99;
rl.lambda        = 0.95; 
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
netQ.nodes          = [netQ.nx  64  64  64 netQ.ny];  
netQ.actFunIdx      = [0        1   4   4   0];
netQ.actBound       = [1        1   1   1   1];
% Observations standard deviation
netQ.sv             = 2*ones(1,1);   
netQ.svDecayFactor  = 0.9999;
netQ.svmin          = 0.3;
% Parameter initialization | {Xavier, He}
netQ.initParamType  = 'He';
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
netA.nodes          = [netA.nx  64  64 netA.ny];  
netA.actFunIdx      = [0        4   4   1];
netA.actBound       = [1        1   1   1];
% Observations standard deviation
netA.sv             = 2*ones(1,1);    
netA.svDecayFactor  = 0.9999;
netA.svmin          = 0.3;
% Parameter initialization | {Xavier, He}
netA.initParamType  = 'He';

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


