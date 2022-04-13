%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         example_1D
% Description:  1D optimization problem
% Authors:      Luong-Ha Nguyen & James-A. Goulet
% Created:      May 25, 2021
% Updated:      May 25, 2021
% Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
% Copyright (c) 2021 Luong-Ha Nguyen & James-A. Goulet 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
% clc
 close all
set(0,'DefaultAxesFontName','Helvetica')
set(0,'defaultLineLineWidth',2)
set(0,'DefaultAxesFontSize',14)
set(0,'defaulttextfontsize',14)
format shortE
%% Data
rng(1)
modelName  = 'example_min';
dataName   = '1D';
ntrain     = 100;
ntest      = 5;
normalizeData = false;
a           = -1;
b           = 1;

% 
f          = @(x) x.^3 - 3 * x;
df         = @(x) 3 * x.^2 - 3;
cdx        = @(x, Sx) 6 * Sx .* x;


sv         = 0.1;
x          = linspace(-2, 2, ntrain);
Sx         = .001*ones(ntrain, 1);
xtrain     = [x(1:end-3), 1, -1,2]';
xtrainOrg  = xtrain;

xtest      = [0 0 0 0 0]';
ytrainTrue = f(xtrain);
ytrain     = f(xtrain) + normrnd(0, sv, [ntrain, 1]);
ytestTrue  = f(xtest);
ytest      = ytestTrue + normrnd(0, sv, [ntest, 1]);
nx         = size(xtrain, 2);
ny         = size(ytrain, 2);
dytrain    = df(xtrain);
dytest     = df(xtest);
Cdxtrain   = cdx(xtrain, Sx);

[~, idx] = sort(xtrain);
% plot(xtrain(idx), ytrain(idx));
maxXtrain = max(xtrain, [], 1);
minXtrain = min(xtrain, [], 1);
maxYtrain = max(ytrain);
minYtrain = min(ytrain);
if normalizeData    
    x = [normalize([xtrain],'range', [-1 1]); xtest];
    y = [normalize([ytrain],'range', [-1 1]); ytest];
    ytrain = y(1:length(ytrain), :);
    ytest  = y(length(ytrain)+1:end);
    xtrain = x(1:length(xtrain), :);
    xtest  = x(length(xtrain)+1:end, :);  
    Cx     = (maxXtrain - minXtrain) ./ (b - a);
    Cy     = (maxYtrain - minYtrain) ./ (b - a);
    dytrain =  dytrain * Cx / Cy;
%     Sx    = 0.01*abs(xtrain);
    Cdxtrain = Cdxtrain  .* (Cx.^2) / Cy  + 0*xtrain;    
end
% plot(xtrain(idx), ytrain(idx))
%% Net
% GPU 1: yes; 0: no
net.task           = 'regression';
net.modelName      = modelName;
net.dataName       = dataName;
net.cd             = cd;
net.saveModel      = 1;
net.maxEpoch       = 100;  
% GPU
net.gpu            = false;
net.cuda           = false;
net.numDevices     = 1;
% Data type object half or double precision
net.dtype          = 'single';
% Number of input covariates
net.nx             = nx; 
% Number of output responses
net.nl             = ny;
net.nv2            = ny;
net.ny             = 1*ny; 
% Batch size 
net.batchSize      = 1; 
net.repBatchSize   = 1; 
% Layer| 1: FC; 2:conv; 3: max pooling; 
% 4: avg pooling; 5: layerNorm; 6: batchNorm 
% Activation: 1:tanh; 2:sigm; 3:cdf; 4:relu; 5:softplus
net.layer          = [1         1   1   1];
net.nodes          = [net.nx    64  64  net.ny]; 
net.actFunIdx      = [0         1   4   0];
net.actBound       = [1         1   1   1];
% Observations standard deviation
net.sv             = 0.1 * ones(1, 1, net.dtype);  
net.noiseType      = 'none';
% Parameter initialization
net.initParamType  = 'He';
net.gainSw         = 1*ones(1, length(net.layer) - 1);
net.gainMw         = ones(1, length(net.layer) - 1);
net.gainSb         = ones(1, length(net.layer) - 1);
net.gainMb         = ones(1, length(net.layer) - 1);
% Last layer update
net.wxupdate        = true;
net.collectDev      = true;
net.convariateEstm  = true;
net.errorRateEval   = true;

%% Initial point
net.signOpt = 1; % - : min; + : max
x0          = -0.25 * ones(net.nx, 1, net.dtype);
Sx0         = (0.001) * ones(net.nx, 1, net.dtype);

%% Run
if net.gpu
    xtrain = gpuArray(cast(xtrain, net.dtype));
    ytrain = gpuArray(cast(ytrain, net.dtype));
    xtest  = gpuArray(cast(xtest, net.dtype));
    ytest  = gpuArray(cast(ytest, net.dtype));
    x0     = gpuArray(x0);
    Sx0    = gpuArray(Sx0);
%     net.sv = gpuArray(net.sv);
end
task.runDerivatveEvaluation(net, xtrain, ytrain, dytrain, Cdxtrain, xtest, ytest, dytest, Sx, ytrainTrue)
%[xopt1]=task.runOptimization(net, xtrain, ytrain, xtest, ytest, 0, 1, x0, Sx0); % simultaneously Layer-wise opt
if normalizeData
    xopt = (xopt1 - a) / (b - a) .* (maxXtrain' - minXtrain') + minXtrain'
end


