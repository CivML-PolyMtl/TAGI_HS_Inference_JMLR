%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         network
% Description:  Build networks relating to each task (task.m)
% Authors:      Luong-Ha Nguyen & James-A. Goulet
% Created:      July 02, 2020
% Updated:      August 26, 2021
% Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
% Copyright (c) 2021 Luong-Ha Nguyen & James-A. Goulet. All rights reserved. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
classdef network
    methods (Static)
        % Reinforcement learning      
        function [dthetaQ, normStatQ, malQ] = tagiOptNstepQ(netQ, netA,...
                state, action, mr, Sr)
            actFunIdx = 1;
            [thetaQ, statesQ, normStatQ, maxIdxQ] = network.extractNet(netQ); 
            
            % Get Q values 
            [malA] = act.meanVar(action, action, zeros(size(action)), ...
                actFunIdx, netQ.actBound(1), netQ.batchSize, ...
                netQ.repBatchSize, netQ.gpu) ;  
            ma0Q = tagi.attachMeanVar(state, malA, netQ.nx-netA.ny, netA.ny,...
                netQ.batchSize, netQ.repBatchSize);
            
            statesQ = tagi.initializeInputs(statesQ, ma0Q, [], [], [], [],...
                [], [], [], [], netQ.xsc);
            
            [statesQ, normStatQ, maxIdxQ] = tagi.feedForwardPassCUDA(netQ,...
                thetaQ, normStatQ, statesQ, maxIdxQ);
            [~, ~, maQ] = tagi.extractStates(statesQ);
            malQ = maQ{end};
            
            % Update netQ
            [deltaMQ, deltaSQ, deltaMxQ,...
                deltaSxQ] = tagi.hiddenStateBackwardPassCUDA(netQ, thetaQ,...
                normStatQ, statesQ, mr, Sr, [], maxIdxQ);
            dthetaQ = tagi.parameterBackwardPassCUDA(netQ, thetaQ, normStatQ,...
                statesQ, deltaMQ, deltaSQ, deltaMxQ, deltaSxQ);              
        end      
        function [dthetaA, numOppSign] = tagiOptA(netQ, netA, state)
            [thetaQ, statesQ, normStatQ, maxIdxQ] = network.extractNet(netQ);
            [thetaA, statesA, normStatA, maxIdxA] = network.extractNet(netA);
            zeroPadS = zeros(size(state), 'like', state);
            dlayerQ = 1;
            
            % Get Q values                    
            statesA = tagi.initializeInputs(statesA, state, [], [], [], [],...
                [], [], [], [], netA.xsc);
            [statesA, normStatA, maxIdxA] = tagi.feedForwardPassCUDA(netA,...
                thetaA, normStatA, statesA, maxIdxA);
            [~, ~, maA, SaA] = tagi.extractStates(statesA); 
            
            mz0Q = tagi.attachMeanVar(state, maA{end}, netQ.nx - netA.ny,...
                netA.ny, netQ.batchSize, netQ.repBatchSize);
            Sz0Q = tagi.attachMeanVar(zeroPadS, SaA{end}, netQ.nx - netA.ny,...
                netA.ny, netQ.batchSize, netQ.repBatchSize);       
                        
            statesQ = tagi.initializeInputs(statesQ, mz0Q, Sz0Q, [], [], [],...
                [], [], [], [], netQ.xsc);
            
            [statesQ, normStatQ, maxIdxQ,...
                mdaQ, SdaQ] = tagi.feedForwardPassCUDA(netQ, thetaQ, normStatQ,...
                statesQ, maxIdxQ);
            [mdgQ, SdgQ, CdgzQ] = tagi.derivative(netQ, thetaQ, normStatQ,...
                statesQ, mdaQ, SdaQ, dlayerQ);
            
            % Update optimal action
            [~, mdglA]  = tagi.detachMeanVar(mdgQ{dlayerQ}, netQ.nx - netA.ny,...
                netA.ny, netQ.batchSize, netQ.repBatchSize);
            [~, SdglA]  = tagi.detachMeanVar(SdgQ{dlayerQ}, netQ.nx - netA.ny,...
                netA.ny, netQ.batchSize, netQ.repBatchSize);
            [~, CdgzlA] = tagi.detachMeanVar(CdgzQ{dlayerQ}, netQ.nx - netA.ny,...
                netA.ny, netQ.batchSize, netQ.repBatchSize);
     
            [deltaMzlA, deltaSzlA] = tagi.fowardHiddenStateUpdate(mdglA, ...
                SdglA, CdgzlA, 0, netA.gpu);
            idxSign    = sign(mdglA) ~= sign(deltaMzlA);
            numOppSign = sum(idxSign);
            deltaMzlA  = sign(mdglA) .* abs(deltaMzlA);            
            
            % Update netA
            malA = maA{end} + deltaMzlA;
            SalA = SaA{end} + deltaSzlA;
            
            [deltaMA, deltaSA, deltaMxA,...
                deltaSxA] = tagi.hiddenStateBackwardPassCUDA(netA, thetaA,...
                normStatA, statesA, malA, SalA, [], maxIdxA);
            dthetaA = tagi.parameterBackwardPassCUDA(netA, thetaA, normStatA,...
                statesA, deltaMA, deltaSA, deltaMxA, deltaSxA);
        end 
        
        function [thetaP, normStatP] = DQN1netCUDA(netP, thetaP, thetaT,...
                statesP, normStatP, maxIdxP, state, action, nextState,...
                reward, finalState)
            % Get updated indices for last layer
            udIdx = dp.selectIndices(action, ...
                netP.batchSize * netP.repBatchSize, ...
                netP.rl.numActions, netP.dtype);    
            
            % Get next Q values
            statesT = tagi.initializeInputs(statesP, nextState, ...
                nextState .* 0, [], [], [], [], [], [], [], netP.xsc);                        
            statesT = tagi.feedForwardPassCUDA(netP, thetaT, normStatP,...
                statesT, maxIdxP);               
            [~, ~, nextMq, nextSq] = tagi.extractStates(statesT);
            nextMq = reshape(nextMq{end}, [netP.ny, ...
                netP.batchSize * netP.repBatchSize]);
            nextSq = reshape(nextSq{end}, [netP.ny, ...
                netP.batchSize * netP.repBatchSize]);  
            [nextMq, nextSq] = rltagi.nextQvalues(gather(nextMq), ...
                gather(nextSq), netP.ny);
            nextMq(finalState) = zeros(1, 1, 'like', nextMq);
            nextSq(finalState) = zeros(1, 1, 'like', nextSq);    
            
            % Get current Q
            netP.trainMode = true;
            statesP = tagi.initializeInputs(statesP, state, state .* 0, [],...
                [], [], [], [], [], [], netP.xsc);
            [statesP, normStatP, maxIdxP] = tagi.feedForwardPassCUDA(netP,...
                thetaP, normStatP, statesP, maxIdxP);
            
            % Get observation
            y  = nextMq * netP.rl.gamma + reward;
            Sy = nextSq * (netP.rl.gamma^2);      
            
            % Update policy network
            [deltaM, deltaS, deltaMx,...
                deltaSx] = tagi.hiddenStateBackwardPassCUDA(netP, thetaP,...
                normStatP, statesP, y, Sy, udIdx, maxIdxP);
            dthetaP = tagi.parameterBackwardPassCUDA(netP, thetaP, ...
                normStatP, statesP, deltaM, deltaS, deltaMx, deltaSx);
            thetaP  = tagi.globalParameterUpdate(thetaP, dthetaP, netP.gpu);
        end
        function [deltaThetaP, normStatP, malP, SalP] = nstepQ1net(netP,...
                thetaP, normStatP, statesP, maxIdxP, state, action, mr, Sr) 
            if netP.gpu
                state  = gpuArray(single(state));
                mr = gpuArray(single(mr));
                Sr = gpuArray(single(Sr));
            end  
            
            % Get Q values from policy network (netP)
            statesP = tagi.initializeInputs(statesP, state, [], [], [], [],...
                [], [], [], [], netP.xsc);
            [statesP, normStatP, maxIdxP] = tagi.feedForwardPass(netP, ...
                thetaP, normStatP, statesP, maxIdxP);
            udIdx = dp.selectIndices(action, ...
                netP.batchSize * netP.repBatchSize, netP.rl.numActions,...
                netP.dtype); 
            [~, ~, maP, SaP] = tagi.extractStates(statesP);
            malP = maP{end}(udIdx);
            SalP = SaP{end}(udIdx);
            
            % Update netP   
            [deltaMP, deltaSP, deltaMxP,...
                deltaSxP] = tagi.hiddenStateBackwardPass(netP, thetaP,...
                normStatP, statesP, mr, Sr, udIdx, maxIdxP);
            deltaThetaP = tagi.parameterBackwardPass(netP, thetaP,...
                normStatP, statesP, deltaMP, deltaSP, deltaMxP, deltaSxP);            
        end
        function [deltaThetaP, normStatP, malP, SalP] = nstepQ1netCUDA(netP,...
                thetaP, normStatP, statesP, maxIdxP, state, action, mr, Sr) 
            if netP.gpu
                state  = gpuArray(single(state));
                mr     = gpuArray(single(mr));
                Sr     = gpuArray(single(Sr));
                action = gpuArray(single(action));
            end  
            
            % Get Q values from policy network (netP)
            statesP = tagi.initializeInputs(statesP, state, [], [], [], [],...
                [], [], [], [], netP.xsc);
            [statesP, normStatP, maxIdxP] = tagi.feedForwardPassCUDA(netP,...
                thetaP, normStatP, statesP, maxIdxP);
            udIdx = dp.selectIndices(action, ...
                netP.batchSize * netP.repBatchSize, netP.rl.numActions, netP.dtype);
            [~, ~, maP, SaP] = tagi.extractStates(statesP);
            malP = maP{end}(udIdx);
            SalP = SaP{end}(udIdx);
            
            % Update netP   
            [deltaMP, deltaSP, deltaMxP, ...
                deltaSxP] = tagi.hiddenStateBackwardPassCUDA(netP, thetaP,...
                normStatP, statesP, mr, Sr, action, maxIdxP);
            deltaThetaP = tagi.parameterBackwardPassCUDA(netP, thetaP,...
                normStatP, statesP, deltaMP, deltaSP, deltaMxP, deltaSxP);            
        end   
               
        % Adversarial attacks
        function [advImg, advlabels] = adversarialGeneration(net, theta,...
                states, normStat, maxIdx, imdb, Sx, classObs, classIdx, Niter)
            if net.learnSv == 1
                ny = net.ny - net.nv2;
            else
                ny = net.ny;
            end
            numDataPerBatch = net.batchSize * net.repBatchSize;
            advImgloop   = cell((net.numClasses - 1) * Niter, 1);
            advlabelloop = cell((net.numClasses - 1) * Niter , 1);
            if net.gpu
                advImgloop(:) = {zeros(1, 1, net.dtype, 'gpuArray')};
                classObs      = gpuArray(cast(classObs, net.dtype));
                classIdx      = gpuArray(cast(classIdx, net.dtype));
            else
                advImgloop(:) = {zeros(1, 1, net.dtype)};
            end
            
            loop = 0;
            for i = 1 : Niter
                % Take real samples
                idxBatch = numDataPerBatch * (i - 1) + 1 : numDataPerBatch * i;
                [xloop, ~, truelabels] = dp.imagenetDataLoader(idxBatch,...
                    imdb, net.da, net.imgSize, net.resizedSize,...
                    net.nx, ny, net.nye, net.batchSize, net.repBatchSize,...
                    net.dtype, net.trainMode, net.resizeMode, net.gpu);
                
                % Create adversarial examples
                Sxloop      = xloop .* 0 + Sx;
                advlabeltmp = repmat([0 : net.numClasses - 1]', ...
                    [1, numDataPerBatch]);
                advlabeltmp(advlabeltmp == truelabels') = [];
                advlabeltmp = reshape(advlabeltmp, ...
                    [net.numClasses - 1, numDataPerBatch]);
                
                for c = 1: net.numClasses - 1
                    loop  = loop + 1;
                    yadv  = classObs(advlabeltmp(c, :) + 1, :)';
                    udIdx = classIdx(advlabeltmp(c, :) + 1, :)';
                    udIdx = dp.selectIndices(udIdx', numDataPerBatch, ny,...
                        net.dtype);
                    yadv  = yadv(:);
                    udIdx = udIdx(:);
                    
                    for e = 1 : net.maxEpoch
                        states = tagi.initializeInputs(states, xloop,...
                            Sxloop, [], [], [], [], [], [], [], net.xsc);
                        % Feed to network
                        [states, normStat, maxIdx] = tagi.feedForwardPass(net,...
                            theta, normStat, states, maxIdx);
                        % Update hidden states
                        [~, ~, ~, ~, deltaMz0, ...
                            deltaSz0] = tagi.hiddenStateBackwardPass(net, ...
                            theta, normStat, states, yadv, [], udIdx, maxIdx);
                        xloop  = xloop + deltaMz0;
                        Sxloop = Sxloop + deltaSz0;
                    end
                    advImgloop{loop}   = xloop;
                    advlabelloop{loop} = advlabeltmp(c, :)';
                    Z = gather(reshape(gather(xloop), ...
                        [prod(net.imgSize), numDataPerBatch])');
                end
            end
            advImg    = cat(1, advImgloop{:, 1});
            advlabels = cat(1, advlabelloop{:, 1});
        end
        function [Pr, sr, erG] = adversarialAttackRate(net, theta, states,...
                normStat, maxIdx, advImg, advlabels, truelabels, classObs,...
                classIdx, Niter)
            numDataPerBatch = net.batchSize * net.repBatchSize;
            numObs = size(advlabels, 1);
            Pr     = zeros(numObs, net.numClasses, net.dtype);
            sr     = zeros(numObs, 1, net.dtype);
            erG    = sr;
            for i = 1 : Niter
                idxBatch = numDataPerBatch * (i - 1) + 1 : numDataPerBatch * i;  
                % Adversarial labels
                al = advlabels(idxBatch);               
                % True labels
                tl = truelabels(idxBatch);                                              
                % Inputs
                imgIdx = numDataPerBatch * prod(net.imgSize) * ...
                    (i - 1) + 1 : numDataPerBatch * prod(net.imgSize) * i; 
                xloop  = advImg(imgIdx);
                Z = gather(reshape(gather(xloop), ...
                    [prod(net.imgSize), numDataPerBatch])');
                states = tagi.initializeInputs(states, xloop, [], [], [],...
                    [], [], [], [], [], net.xsc);                    
                states = tagi.feedForwardPass(net, theta, normStat, states,...
                    maxIdx);
                [~, ~, ma, Sa] = tagi.extractStates(states);
                % Outputs
                ml = reshape(ma{end}, [numDataPerBatch * net.nl, 1]);
                Sl = reshape(Sa{end}, [numDataPerBatch * net.nl, 1]);
                
                % Compute attack success rate
                P = dp.obs2class(ml, Sl, classObs, classIdx);
                P = reshape(P, [net.numClasses, numDataPerBatch])';
                P = gather(P);
                Pr(idxBatch, :)  = P;
                sr(idxBatch, :)  = mt.errorRate(al', P');
                erG(idxBatch, :) = mt.errorRate(tl', P');                
            end
        end
        
        function [advImg, advlabels] = adversarialGenerationCUDA4plot(net,...
                theta, states, normStat, maxIdx, imdb, Sx, classObs,...
                classIdx, Niter)
            if net.learnSv == 1
                ny = net.ny - net.nv2;
            else
                ny = net.ny;
            end
            numDataPerBatch = net.batchSize * net.repBatchSize;
            advImgloop   = cell(net.numClasses, net.numClasses);
            advlabelloop = cell(net.numClasses, net.numClasses);
            if net.gpu
                classObs = gpuArray(cast(classObs, net.dtype));
                classIdx = gpuArray(cast(classIdx, net.dtype));
            end
                      
            
            loop = 0;
            for i = 1 : Niter
                % Take real samples
                idxBatch = numDataPerBatch * (i - 1) + ...
                    1 : numDataPerBatch * i;
                [xloopref, ~, truelabels] = dp.imagenetDataLoader(idxBatch,...
                    imdb, net.da, net.imgSize, net.resizedSize,...
                    net.nx, ny, net.nye, net.batchSize, net.repBatchSize,...
                    net.dtype, net.trainMode, net.resizeMode, net.gpu);
                
                % Create adversarial examples
                advlabeltmp = repmat([0 : net.numClasses - 1]', ...
                    [1, numDataPerBatch]);
                advlabeltmp(advlabeltmp == truelabels') = [];
                advlabeltmp = reshape(advlabeltmp, [net.numClasses - 1, ...
                    numDataPerBatch]);
                Z = reshape(xloopref, [net.imgSize numDataPerBatch]) ...
                    .* net.imgStat(:,:,:, 2) + net.imgStat(:,:,:, 1);
                Z = gather(reshape(gather(Z), [prod(net.imgSize), numDataPerBatch])');
                for j = 1 : numDataPerBatch
                    advImgloop{truelabels(j)+1, truelabels(j)+1} = Z(j, :)';
                end
                for c = 1: net.numClasses - 1
                    loop  = loop + 1;
                    yadv  = classObs(advlabeltmp(c, :) + 1, :)';
                    udIdx = classIdx(advlabeltmp(c, :) + 1, :)';
                    xloop = xloopref;
                    Sxloop = xloopref * 0 + Sx;%(0.01 * abs(xloop));%xloop * 0 + Sx;%(0.2 * abs(xloop)).^2 ;%xloop * 0 + Sx; %0.01 * abs(xloop);
                    
                    for e = 1 : net.maxEpoch
                        states = tagi.initializeInputs(states, xloop,...
                            Sxloop, [], [], [], [], [], [], [], net.xsc);
                        % Feed to network
                        [states, normStat,...
                            maxIdx] = tagi.feedForwardPassCUDA(net, theta,...
                            normStat, states, maxIdx);
                        % Update hidden states
                        [~, ~, ~, ~, deltaMz0,...
                            deltaSz0] = tagi.hiddenStateBackwardPassCUDA(net,...
                            theta, normStat, states, yadv, [], udIdx, maxIdx);
                        xloop  = xloop + deltaMz0;
                        Sxloop = Sxloop + deltaSz0;
                    end
                    advlabelloop{loop} = advlabeltmp(c, :)';
                    Z = reshape(xloop, [net.imgSize numDataPerBatch]) ...
                        .* net.imgStat(:,:,:, 2) + net.imgStat(:,:,:, 1);
                    Z = gather(reshape(gather(Z), ...
                        [prod(net.imgSize), numDataPerBatch])');
                    for j = 1 : numDataPerBatch
                        advImgloop{advlabeltmp(c, j)+1, truelabels(j)+1} = Z(j, :)';
                    end
                end
            end
            advImg    = cell2mat(advImgloop);
            advImg    = reshape(advImg, [prod(net.imgSize), ...
                numDataPerBatch * (net.numClasses)])';
            advlabels = cat(1, advlabelloop{:, 1});
        end
        function [advImg, advlabels] = adversarialGenerationCUDA(net, theta,...
                states, normStat, maxIdx, imdb, Sx, classObs, classIdx, Niter)
            if net.learnSv == 1
                ny = net.ny - net.nv2;
            else
                ny = net.ny;
            end
            numDataPerBatch = net.batchSize * net.repBatchSize;
            advImgloop   = cell((net.numClasses - 1) * Niter, 1);
            advlabelloop = cell((net.numClasses - 1) * Niter , 1);
            if net.gpu
                advImgloop(:) = {zeros(1, 1, net.dtype, 'gpuArray')};
                classObs      = gpuArray(cast(classObs, net.dtype));
                classIdx      = gpuArray(cast(classIdx, net.dtype));
            else
                advImgloop(:) = {zeros(1, 1, net.dtype)};
            end
            
            loop = 0;
            for i = 1 : Niter
                % Take real samples
                idxBatch = numDataPerBatch * (i - 1) +...
                    1 : numDataPerBatch * i;
                [xloopref, ~, truelabels] = dp.imagenetDataLoader(idxBatch,...
                    imdb, net.da, net.imgSize, net.resizedSize,...
                    net.nx, ny, net.nye, net.batchSize, net.repBatchSize,...
                    net.dtype, net.trainMode, net.resizeMode, net.gpu);
                
                % Create adversarial examples
                advlabeltmp = repmat([0 : net.numClasses - 1]', ...
                    [1, numDataPerBatch]);
                advlabeltmp(advlabeltmp == truelabels') = [];
                advlabeltmp = reshape(advlabeltmp, ...
                    [net.numClasses - 1, numDataPerBatch]);
                Z = reshape(xloopref, [net.imgSize numDataPerBatch]) ...
                    .* net.imgStat(:,:,:, 2) + net.imgStat(:,:,:, 1);
                Z = gather(reshape(gather(Z), ...
                    [prod(net.imgSize), numDataPerBatch])');
                for c = 1: net.numClasses - 1
                    loop  = loop + 1;
                    yadv  = classObs(advlabeltmp(c, :) + 1, :)';
                    udIdx = classIdx(advlabeltmp(c, :) + 1, :)';
                    xloop  = xloopref;
                    Sxloop = xloopref * 0 + Sx;%(0.01 * abs(xloop));%xloop * 0 + Sx;%(0.2 * abs(xloop)).^2 ;%xloop * 0 + Sx; %0.01 * abs(xloop);
                    
                    for e = 1 : net.maxEpoch
                        states = tagi.initializeInputs(states, xloop, ...
                            Sxloop, [], [], [], [], [], [], [], net.xsc);
                        % Feed to network
                        [states, normStat, ...
                            maxIdx] = tagi.feedForwardPassCUDA(net, theta,...
                            normStat, states, maxIdx);
                        % Update hidden states
                        [~, ~, ~, ~, deltaMz0, ...
                            deltaSz0] = tagi.hiddenStateBackwardPassCUDA(net,...
                            theta, normStat, states, yadv, [], udIdx, maxIdx);
                        xloop  = xloop + deltaMz0;
                        Sxloop = Sxloop + deltaSz0;
                    end
                    advImgloop{loop}   = xloop;
                    advlabelloop{loop} = advlabeltmp(c, :)';
                    Z = reshape(xloop, [net.imgSize numDataPerBatch]) ...
                        .* net.imgStat(:,:,:, 2) + net.imgStat(:,:,:, 1);
                    Z = gather(reshape(gather(Z), ...
                        [prod(net.imgSize), numDataPerBatch])');
                end
            end
            advImg    = cat(1, advImgloop{:, 1});
            advlabels = cat(1, advlabelloop{:, 1});
        end
        function [Pr, sr, erG] = adversarialAttackRateCUDA(net, theta,...
                states, normStat, maxIdx, advImg, advlabels, truelabels,...
                classObs, classIdx, Niter)
            numDataPerBatch = net.batchSize * net.repBatchSize;
            numObs = size(advlabels, 1);
            Pr     = zeros(numObs, net.numClasses, net.dtype);
            sr     = zeros(numObs, 1, net.dtype);
            erG    = sr;
            for i = 1 : Niter
                idxBatch = numDataPerBatch * (i - 1) +...
                    1 : numDataPerBatch * i;  
                % Adversarial labels
                al = advlabels(idxBatch);               
                % True labels
                tl = truelabels(idxBatch);                                              
                % Inputs
                imgIdx = numDataPerBatch * prod(net.imgSize) * (i - 1) +...
                    1 : numDataPerBatch * prod(net.imgSize) * i; 
                xloop  = advImg(imgIdx);
                Z      = reshape(xloop, [net.imgSize numDataPerBatch]) ...
                    .* net.imgStat(:,:,:, 2) + net.imgStat(:,:,:, 1);
                Z      = gather(reshape(gather(Z(:)), ...
                    [prod(net.imgSize), numDataPerBatch])');
                states = tagi.initializeInputs(states, xloop, [], [], [],...
                    [], [], [], [], [], net.xsc);                    
                states = tagi.feedForwardPassCUDA(net, theta, normStat,...
                    states, maxIdx);
                [~, ~, ma, Sa] = tagi.extractStates(states);
                % Outputs
                ml = reshape(ma{end}, [numDataPerBatch * net.nl, 1]);
                Sl = reshape(Sa{end}, [numDataPerBatch * net.nl, 1]);
                
                % Compute attack success rate
                P = dp.obs2class(ml, Sl, classObs, classIdx);
                P = reshape(P, [net.numClasses, numDataPerBatch])';
                P = gather(P);
                Pr(idxBatch, :)  = P;
                sr(idxBatch, :)  = mt.errorRate(al', P');
                erG(idxBatch, :) = mt.errorRate(tl', P');                
            end
        end
        
        % Derivative net
        function [theta, normStat, mzl, Szl, mdg,...
                Sdg, Cdgz] = batchDerivativeCheck(net, theta, normStat,...
                states, maxIdx, x, Sx, y, dlayer)
            % Initialization
            numObs = size(x, 1);
            numDataPerBatch = net.repBatchSize*net.batchSize;
            mzl  = zeros(numObs, net.ny, net.dtype);
            Szl  = zeros(numObs, net.ny, net.dtype);
            mdg  = zeros(net.nodes(dlayer), net.nx, net.dtype);
            Sdg  = mdg;
            Cdgz = mdg;
            % Loop
            loop = 0;
            for i = 1:numDataPerBatch:numObs
                loop     = loop + 1; 
                if numDataPerBatch==1
                    idxBatch = i : i + net.batchSize - 1;
                else
                    idxBatch = i : i + numDataPerBatch - 1;
                end
                % Covariate            
                xloop  = reshape(x(idxBatch, :)', ...
                    [net.batchSize * net.nx, net.repBatchSize]);
                Sxloop = reshape(Sx(idxBatch, :)', ...
                    [net.batchSize * net.nx, net.repBatchSize]);
                states = tagi.initializeInputs(states, xloop, Sxloop, [],...
                    [], [], [], [], [], [], net.xsc);
         
                % Observation
                if net.trainMode
                    yloop = reshape(y(idxBatch, :)', ...
                        [net.batchSize * net.nl, net.repBatchSize]); 
                    [states, normStat, maxIdx,...
                        mda, Sda] = tagi.feedForwardPass(net, theta, ...
                        normStat, states, maxIdx);
                    [mdgi, Sdgi, Cdgzi] = tagi.derivative(net, theta,...
                        normStat, states, mda, Sda, dlayer);
                    [deltaM, deltaS,deltaMx,...
                        deltaSx] = tagi.hiddenStateBackwardPass(net, theta,...
                        normStat, states, yloop, [], [], maxIdx);
                    dtheta = tagi.parameterBackwardPass(net, theta,...
                        normStat, states, deltaM, deltaS, deltaMx, deltaSx);
                    theta  = tagi.globalParameterUpdate(theta, dtheta,...
                        net.gpu); 
                else
                    [states, normStat, maxIdx,...
                        mda, Sda] = tagi.feedForwardPass(net, theta,...
                        normStat, states, maxIdx);
                    [mdgi, Sdgi, Cdgzi] = tagi.derivative(net, theta,...
                        normStat, states, mda, Sda, dlayer);
                end
                [~, ~, ma, Sa]    = tagi.extractStates(states);
                mzl(idxBatch, :)  = gather(reshape(ma{end}, ...
                    [net.ny, numDataPerBatch])');
                Szl(idxBatch, :)  = gather(reshape(Sa{end}, ...
                    [net.ny, numDataPerBatch])');  
                mdg(idxBatch, :)  = gather(reshape(mdgi{dlayer}, ...
                    [net.nodes(dlayer), numDataPerBatch])');
                Sdg(idxBatch, :)  = gather(reshape(Sdgi{dlayer}, ...
                    [net.nodes(dlayer), numDataPerBatch])');
                Cdgz(idxBatch, :) = gather(reshape(Cdgzi{dlayer}, ...
                    [net.nodes(dlayer), numDataPerBatch])');
            end
        end
        function [theta, normStat, mzl, Szl, mdg, Sdg, Cdgz,...
                xopt, Sxopt] = optimizationCUDA(net, theta, normStat,...
                states, maxIdx,  netO, normStatO, statesO, maxIdxO, x, Sx,...
                y, lastLayer, dlayer, xopt, Sxopt)
            % Initialization
            numObs = size(x, 1);
            numDataPerBatch = net.repBatchSize * net.batchSize;
            mzl  = zeros(numObs, net.ny, net.dtype);
            Szl  = zeros(numObs, net.ny, net.dtype);
            mdg  = zeros(net.nodes(dlayer), net.nx, net.dtype);
            Sdg  = mdg;
            Cdgz = mdg;
            % Loop
            loop = 0;
            for i = 1 : numDataPerBatch : numObs
                loop     = loop + 1; 
                if numDataPerBatch==1
                    idxBatch = i: i + net.batchSize - 1;
                else
                    if numObs-i >= numDataPerBatch
                        idxBatch = i:i+numDataPerBatch-1;
                    else
                        idxBatch = [i : numObs, ...
                            randperm(i-1, numDataPerBatch-numObs+i-1)];
                    end
                end
                % Covariate            
                xloop     = reshape(x(idxBatch, :)', ...
                    [net.batchSize * net.nx, net.repBatchSize]);
                Sxloop    = reshape(Sx(idxBatch, :)', ...
                    [net.batchSize * net.nx, net.repBatchSize]);
                states    = tagi.initializeInputs(states, xloop, Sxloop,...
                    [], [], [], [], [], [], [], net.xsc);
                optStates = tagi.initializeInputs(statesO, xopt, Sxopt,...
                    [], [], [], [], [], [], [], netO.xsc);
         
                if net.trainMode
                    yloop = reshape(y(idxBatch, :)', ...
                        [net.batchSize * net.nl, net.repBatchSize]); 
                    [states, normStat, ...
                        maxIdx] = tagi.feedForwardPassCUDA(net, theta, ...
                        normStat, states, maxIdx);
                    [optStates, normStatO, maxIdxO, ...
                        mda, Sda] = tagi.feedForwardPassCUDA(netO, theta, ...
                        normStatO, optStates, maxIdxO);
                    [mdgi, Sdgi, Cdgzi] = tagi.derivative(netO, theta, ...
                        normStatO, optStates, mda, Sda, dlayer);
                    
                    % Update optimal points
                    [deltaMzl, ...
                     deltaSzl] = tagi.fowardHiddenStateUpdate(mdgi{lastLayer},...
                     Sdgi{lastLayer}, Cdgzi{lastLayer}, 0, net.gpu);
                    
                    % Minus sign for minimum and plus sign for maximum
                    deltaMzl = sign(mdgi{lastLayer}) .* abs(deltaMzl)...
                        * net.signOpt;
                    
                    [deltaMzopt, ...
                     deltaSzopt] = tagi.optimizationHiddenStateBackwardPassCUDA(netO, ...
                     theta, normStatO, optStates, deltaMzl, deltaSzl, ...
                     maxIdxO, lastLayer, dlayer);
                    xopt  = xopt + deltaMzopt;
                    Sxopt = Sxopt + deltaSzopt;
                    
                    % Update network
                    [deltaM, deltaS,deltaMx, ...
                        deltaSx] = tagi.hiddenStateBackwardPassCUDA(net,...
                        theta, normStat, states, yloop, [], gpuArray(single(nan)), maxIdx);
                    dtheta = tagi.parameterBackwardPassCUDA(net, theta, ...
                        normStat, states, deltaM, deltaS, deltaMx, deltaSx);
                    theta  = tagi.globalParameterUpdate(theta, dtheta, ...
                        net.gpu); 
                else
                    [states, normStat, maxIdx, ...
                        mda, Sda] = tagi.feedForwardPassCUDA(net, theta, ...
                        normStat, states, maxIdx);
                    [mdgi, Sdgi, Cdgzi] = tagi.derivative(net, theta, ...
                        normStat, states, mda, Sda, dlayer);
                end
                [~, ~, ma, Sa]    = tagi.extractStates(states);
                mzl(idxBatch, :)  = gather(reshape(ma{end}, ...
                    [net.ny, numDataPerBatch])');
                Szl(idxBatch, :)  = gather(reshape(Sa{end}, ...
                    [net.ny, numDataPerBatch])');  
            end
        end
        
        function [theta, normStat, mzl, Szl, mdg, Sdg, Cdgz, ...
                xopt, Sxopt] = optimization(net, theta, normStat, states,...
                maxIdx, netO, normStatO, statesO, maxIdxO, x, Sx, y, ...
                lastLayer, dlayer, xopt, Sxopt)
            % Initialization
            numObs = size(x, 1);
            numDataPerBatch = net.repBatchSize * net.batchSize;
            mzl  = zeros(numObs, net.ny, net.dtype);
            Szl  = zeros(numObs, net.ny, net.dtype);
            mdg  = zeros(net.nodes(dlayer), net.nx, net.dtype);
            Sdg  = mdg;
            Cdgz = mdg;
            % Loop
            loop = 0;
            for i = 1 : numDataPerBatch : numObs
                loop     = loop + 1; 
                if numDataPerBatch==1
                    idxBatch = i : i + net.batchSize - 1;
                else
                    idxBatch = i : i + numDataPerBatch - 1;
                end
                % Covariate            
                xloop     = reshape(x(idxBatch, :)', ...
                    [net.batchSize * net.nx, net.repBatchSize]);
                Sxloop    = reshape(Sx(idxBatch, :)', ...
                    [net.batchSize * net.nx, net.repBatchSize]);
                states    = tagi.initializeInputs(states, xloop, Sxloop,...
                    [], [], [], [], [], [], [], net.xsc);
                optStates = tagi.initializeInputs(statesO, xopt, Sxopt,...
                    [], [], [], [], [], [], [], netO.xsc);
         
                if net.trainMode
                    yloop = reshape(y(idxBatch, :)', ...
                        [net.batchSize*net.nl, net.repBatchSize]); 
                    [states, normStat, maxIdx] = tagi.feedForwardPass(net,...
                        theta, normStat, states, maxIdx);
                    [optStates, normStatO, maxIdxO, ...
                        mda, Sda] = tagi.feedForwardPass(netO, theta, ...
                        normStatO, optStates, maxIdxO);
                    [mdgi, Sdgi, Cdgzi] = tagi.derivative(netO, theta, ...
                        normStatO, optStates, mda, Sda, lastLayer);
                    
                    [deltaMzl, ...
                     deltaSzl] = tagi.fowardHiddenStateUpdate(mdgi{lastLayer},...
                     Sdgi{lastLayer}, Cdgzi{lastLayer}, 0, net.gpu);
                    deltaMzl = sign(mdgi{lastLayer}) .* abs(deltaMzl) ...
                        *  net.signOpt;
                    xopt  = xopt + deltaMzl;
                    Sxopt = Sxopt + deltaSzl;
                    
                    % Update network
                    [deltaM, deltaS,deltaMx, ...
                        deltaSx] = tagi.hiddenStateBackwardPass(net, theta,...
                        normStat, states, yloop, [], [], maxIdx);
                    dtheta = tagi.parameterBackwardPass(net, theta, ...
                        normStat, states, deltaM, deltaS, deltaMx, deltaSx);
                    theta  = tagi.globalParameterUpdate(theta, dtheta, ...
                        net.gpu); 
                else
                    [states, normStat, maxIdx, ...
                        mda, Sda] = tagi.feedForwardPass(net, theta, ...
                        normStat, states, maxIdx);
                    [mdgi, Sdgi, Cdgzi] = tagi.derivative(net, theta, ...
                        normStat, states, mda, Sda, dlayer);
                end
                [~, ~, ma, Sa]   = tagi.extractStates(states);
                mzl(idxBatch, :) = gather(reshape(ma{end}, ...
                    [net.ny, numDataPerBatch])');
                Szl(idxBatch, :) = gather(reshape(Sa{end}, ...
                    [net.ny, numDataPerBatch])');  
            end
        end
        
        % Classification
        function [theta, normStat, Pn, er, sv] = classification(net, theta,...
                normStat, states, maxIdx, imdb, classObs, classIdx, Niter)
            % Initialization           
            numDataPerBatch = net.batchSize*net.repBatchSize;
            if net.learnSv == 1
                ny = net.ny-net.nv2;
            else
                ny = net.ny;
            end                         
            
            % Loop
            loop     = 0;
            time_tot = 0;
                        
            if net.trainMode
                if net.errorRateEval == 1
                    Pn = zeros(Niter*numDataPerBatch, net.numClasses, net.dtype);
                    er = zeros(Niter*numDataPerBatch, 1, net.dtype);
                else
                    er = nan;
                    Pn = nan;
                end
                
                % Training
                for i = 1:Niter
                    timeval  = tic;
                    loop     = loop + 1;
                    idxBatch = numDataPerBatch*(i-1)+1:numDataPerBatch*i;
                    % Get batch of data
                    idx      = randperm(imdb.numImages, numDataPerBatch);
                    [xloop, yloop, labels,...
                        udIdx] = dp.imagenetDataLoader(idx, imdb, net.da,...
                        net.imgSize, net.resizedSize, net.nx, ny, net.nye,...
                        net.batchSize, net.repBatchSize, net.dtype,...
                        net.trainMode, net.resizeMode, net.gpu, net.cuda);
                    states = tagi.initializeInputs(states, xloop, [], [], ...
                        [], [], [], [], [], [], net.xsc);
                     
                    % Feed to network
                    [states, normStat, maxIdx] = tagi.feedForwardPass(net, ...
                        theta, normStat, states, maxIdx);
                    
                    % Update parameters
                    [deltaM, deltaS,deltaMx, ...
                        deltaSx] = tagi.hiddenStateBackwardPass(net, theta,...
                        normStat, states, yloop, [], udIdx, maxIdx);
                    dtheta = tagi.parameterBackwardPass(net, theta, ...
                        normStat, states, deltaM, deltaS, deltaMx, deltaSx);
                    theta  = tagi.globalParameterUpdate(theta, dtheta, ...
                        net.gpu);
                    
                    % Extract hidden states for last layer
                    [~, ~, ma, Sa] = tagi.extractStates(states);
                    if net.learnSv == 1
                        [ml, mv2] = tagi.detachMeanVar(ma{end}, net.nl, ...
                            net.nv2, net.batchSize, net.repBatchSize);
                        [Sl, Sv2] = tagi.detachMeanVar(Sa{end}, net.nl, ...
                            net.nv2, net.batchSize, net.repBatchSize);
                        mv2a = act.expFun(mv2, Sv2, net.gpu);
                        ml   = reshape(ml, [ny * numDataPerBatch, 1]);
                        Sl   = reshape(Sl+mv2a, [ny * numDataPerBatch, 1]);
                    else
                        ml = reshape(ma{end}, [numDataPerBatch * net.nl, 1]);
                        Sl = reshape(Sa{end}, [numDataPerBatch * net.nl, 1]);
                    end
                    time_loop = toc(timeval);
                    time_tot  = time_tot + time_loop;
                    time_rem  = time_tot / i * (Niter - i) / 60;
                    
                    % Error rate
                    if net.errorRateEval == 1
                        P = dp.obs2class(ml, Sl, classObs, classIdx);
                        P = reshape(P, [net.numClasses, numDataPerBatch])';
                        P = gather(P);
                        Pn(idxBatch, :) = P;
                        er(idxBatch, :) = mt.errorRate(labels', P');
                        
                        % Display error rate
                        if (mod(i, net.obsShow) == 0 || i == Niter) ...
                                && i > 1 && net.trainMode ...
                                && net.displayMode == 1
                            formatSpec = ' Iter# %2.0e/%0.0e,  Error rate: %3.1f%%, Time left: %1.0f mins\n';
                            fprintf(formatSpec, i, Niter, 100*mean(er(max(1,i-200):i)), time_rem);
                        end
                    else
                        if mod(i, net.obsShow) == 0 ...
                                && i > 1 ...
                                && net.trainMode == 1 ...
                                &&net.displayMode == 1
                            disp(['     Time left : ' sprintf('%0.2f',time_rem) ' mins'])
                        end
                    end
                    
                    % Set momentum for average running after 1st obs                   
                    net.normMomentum = net.normMomentumRef;                    
                end
            else
                % Testing
                Pn = zeros(Niter*numDataPerBatch, net.numClasses, net.dtype);
                er = zeros(Niter*numDataPerBatch, 1, net.dtype);
                % To be completed
                for i = 1:Niter
                    loop     = loop + 1;
                    % Update prior for parameters
                    
                    idxBatch = numDataPerBatch * (i - 1) +...
                        1 : numDataPerBatch * i;
                    [xloop, ~, labels, ~] = dp.imagenetDataLoader(idxBatch,...
                        imdb, net.da, net.imgSize, net.resizedSize, net.nx,...
                        ny, net.nye, net.batchSize, net.repBatchSize,...
                        net.dtype, net.trainMode, net.resizeMode, net.gpu);
                    states = tagi.initializeInputs(states, xloop, [], [], ...
                        [], [], [], [], [], [], net.xsc);
                    
                    states = tagi.feedForwardPass(net, theta, normStat, ...
                        states, maxIdx);
                    [~, ~, ma, Sa] = tagi.extractStates(states);
                    if net.learnSv == 1
                        [ml, mv2] = tagi.detachMeanVar(ma{end}, net.nl, ...
                            net.nv2, net.batchSize, net.repBatchSize);
                        [Sl, Sv2] = tagi.detachMeanVar(Sa{end}, net.nl, ...
                            net.nv2, net.batchSize, net.repBatchSize);
                        mv2a = act.expFun(mv2, Sv2, net.gpu);
                        ml   = reshape(ml, [ny * numDataPerBatch, 1]);
                        Sl   = reshape(Sl+mv2a, [ny * numDataPerBatch, 1]);
                    else
                        ml = reshape(ma{end}, [numDataPerBatch * net.nl, 1]);
                        Sl = reshape(Sa{end}, [numDataPerBatch * net.nl, 1]);
                    end
                    
                    % Error rate
                    if net.errorRateEval == 1
                        P = dp.obs2class(ml, Sl, classObs, classIdx);
                        P = reshape(P, [net.numClasses, numDataPerBatch])';
                        P = gather(P);
                        Pn(idxBatch, :) = P;
                        er(idxBatch, :) = mt.errorRate(labels', P');                        
                    end
                end
            end
            sv = net.sv;
        end 
        function [theta, normStat, Pn, er, sv] = classificationCUDA(net,...
                theta, normStat, states, maxIdx, imdb, classObs, classIdx,...
                Niter)
            % Initialization           
            numDataPerBatch = net.batchSize * net.repBatchSize;
            if net.learnSv == 1
                ny = net.ny-net.nv2;
            else
                ny = net.ny;
            end                         
            
            % Loop
            loop     = 0;
            time_tot = 0;
                        
            if net.trainMode
                if net.errorRateEval
                    Pn = zeros(Niter * numDataPerBatch, net.numClasses,...
                        net.dtype);
                    er = zeros(Niter * numDataPerBatch, 1, net.dtype);
                else
                    er = nan;
                    Pn = nan;
                end
                
                % Training
                for i = 1:Niter
                    timeval  = tic;
                    loop     = loop + 1;
                    idxBatch = numDataPerBatch * (i - 1) +...
                        1 : numDataPerBatch * i;
                    % Get batch of data
                    idx      = randperm(imdb.numImages, numDataPerBatch);
                    [xloop, yloop, labels,...
                        udIdx] = dp.imagenetDataLoader(idx, imdb, net.da,...
                        net.imgSize, net.resizedSize, net.nx, ny, net.nye,...
                        net.batchSize, net.repBatchSize, net.dtype,...
                        net.trainMode, net.resizeMode, net.gpu, net.cuda);
                    states = tagi.initializeInputs(states, xloop, [], [],...
                        [], [], [], [], [], [], net.xsc);
                     
                    % Feed to network
                    [states, normStat,...
                        maxIdx] = tagi.feedForwardPassCUDA(net, theta,...
                        normStat, states, maxIdx);
                    
                    % Update parameters
                    [deltaM, deltaS,deltaMx,...
                        deltaSx] = tagi.hiddenStateBackwardPassCUDA(net,...
                        theta, normStat, states, yloop, [], udIdx, maxIdx);
                    dtheta = tagi.parameterBackwardPassCUDA(net, theta, ...
                        normStat, states, deltaM, deltaS, deltaMx, deltaSx);
                    theta = tagi.globalParameterUpdateCUDA(theta, dtheta, ...
                        net.wxupdate, net.numParamsPerlayer_2);  
                    
                    % Extract hidden states for last layer
                    [~, ~, ma, Sa] = tagi.extractStates(states);
                    if net.learnSv == 1
                        [ml, mv2] = tagi.detachMeanVar(ma{end}, net.nl,...
                            net.nv2, net.batchSize, net.repBatchSize);
                        [Sl, Sv2] = tagi.detachMeanVar(Sa{end}, net.nl,...
                            net.nv2, net.batchSize, net.repBatchSize);
                        mv2a = act.expFun(mv2, Sv2, net.gpu);
                        ml   = reshape(ml, [ny * numDataPerBatch, 1]);
                        Sl   = reshape(Sl+mv2a, [ny * numDataPerBatch, 1]);
                    else
                        ml = reshape(ma{end}, [numDataPerBatch * net.nl, 1]);
                        Sl = reshape(Sa{end}, [numDataPerBatch * net.nl, 1]);
                    end
                    time_loop = toc(timeval);
                    time_tot  = time_tot + time_loop;
                    time_rem  = time_tot / i * (Niter - i) / 60;
                    
%                     % Error rate
                    if net.errorRateEval
                        P = dp.obs2class(ml, Sl, classObs, classIdx);
                        P = reshape(P, [net.numClasses, numDataPerBatch])';
                        P = gather(P);
                        Pn(idxBatch, :) = P;
                        er(idxBatch, :) = mt.errorRate(labels', P');
                        
                        % Display error rate
                        if (mod(i, net.obsShow) == 0 ...
                            || i == Niter)&& i > 1 ...
                            && net.trainMode ...
                            && net.displayMode == 1
                            formatSpec = ' Iter# %2.0e/%0.0e,  Error rate: %3.1f%%, Time left: %1.0f mins\n';
                            fprintf(formatSpec, i, Niter, ...
                                100 * mean(er(max(1, i - 200) : i)), time_rem);
                        end
                    else
                        if mod(i, net.obsShow) == 0 ...
                                && i >1 ...
                                && net.trainMode == 1 ...
                                && net.displayMode == 1
                            disp(['     Time left : ' sprintf('%0.2f',time_rem) ' mins'])
                        end
                    end
                    
                    % Set momentum for average running after 1st obs                   
                    net.normMomentum = net.normMomentumRef;                    
                end
            else
                % Testing
                Pn = zeros(Niter * numDataPerBatch, net.numClasses, net.dtype);
                er = zeros(Niter * numDataPerBatch, 1, net.dtype);
                % To be completed
                for i = 1:Niter
                    loop     = loop + 1;
                    % Update prior for parameters
                    
                    idxBatch = numDataPerBatch * (i - 1) +...
                        1 : numDataPerBatch * i;
                    [xloop, ~, labels, ~] = dp.imagenetDataLoader(idxBatch,...
                        imdb, net.da, net.imgSize, net.resizedSize, net.nx,...
                        ny, net.nye, net.batchSize, net.repBatchSize,...
                        net.dtype, net.trainMode, net.resizeMode, net.gpu);
                    states = tagi.initializeInputs(states, xloop, [], [],...
                        [], [], [], [], [], [], net.xsc);
                    
                    states = tagi.feedForwardPassCUDA(net, theta, normStat,...
                        states, maxIdx);
                    [~, ~, ma, Sa] = tagi.extractStates(states);
                    if net.learnSv == 1
                        [ml, mv2] = tagi.detachMeanVar(ma{end}, net.nl,...
                            net.nv2, net.batchSize, net.repBatchSize);
                        [Sl, Sv2] = tagi.detachMeanVar(Sa{end}, net.nl,...
                            net.nv2, net.batchSize, net.repBatchSize);
                        mv2a = act.expFun(mv2, Sv2, net.gpu);
                        ml   = reshape(ml, [ny * numDataPerBatch, 1]);
                        Sl   = reshape(Sl+mv2a, [ny * numDataPerBatch, 1]);
                    else
                        ml = reshape(ma{end}, [numDataPerBatch * net.nl, 1]);
                        Sl = reshape(Sa{end}, [numDataPerBatch * net.nl, 1]);
                    end
                    
                    % Error rate
                    if net.errorRateEval == 1
                        P = dp.obs2class(ml, Sl, classObs, classIdx);
                        P = reshape(P, [net.numClasses, numDataPerBatch])';
                        P = gather(P);
                        Pn(idxBatch, :) = P;
                        er(idxBatch, :) = mt.errorRate(labels', P');                        
                    end
                end
            end
            sv = net.sv;
        end         
        
        % Regression 
        function [theta, normStat, zl, Szl, sv] = regression(net, theta,...
                normStat, states, maxIdx, x, y)
            % Initialization
            numObs = size(x, 1);
            numDataPerBatch = net.repBatchSize * net.batchSize;
            zl  = zeros(numObs, net.ny, net.dtype);
            Szl = zeros(numObs, net.ny, net.dtype);
            % Loop
            loop = 0;
            for i = 1 : numDataPerBatch : numObs
                loop     = loop + 1; 
                if numDataPerBatch==1
                    idxBatch = i : i + net.batchSize - 1;
                else
                    if numObs - i >= numDataPerBatch
                        idxBatch = i : i + numDataPerBatch - 1;
                    else
                        idxBatch = [i : numObs, randperm(i - 1, ...
                            numDataPerBatch - numObs + i - 1)];
                    end
                end
                % Covariate
                xloop  = reshape(x(idxBatch, :)', ...
                    [net.batchSize * net.nx, net.repBatchSize]);
                states = tagi.initializeInputs(states, xloop, [], [], [],...
                    [], [], [], [], [], net.xsc);
                % Training
                if net.trainMode                 
                    % Observation
                    yloop = reshape(y(idxBatch, :)', ...
                        [net.batchSize * net.nl, net.repBatchSize]);                  
                    [states, normStat, maxIdx] = tagi.feedForwardPass(net,...
                        theta, normStat, states, maxIdx); 
                    [deltaM, deltaS,deltaMx, deltaSx,...
                        ~, ~, sv] = tagi.hiddenStateBackwardPass(net, theta,...
                        normStat, states, yloop, [], [], maxIdx);
                    net.sv = sv;
                    dtheta = tagi.parameterBackwardPass(net, theta, normStat,...
                        states, deltaM, deltaS, deltaMx, deltaSx);
                    theta  = tagi.globalParameterUpdate(theta, dtheta,...
                        net.gpu);  
                    [~, ~, ma, Sa]   = tagi.extractStates(states);
                % Testing    
                else 
                    [states, normStat, maxIdx] = tagi.feedForwardPass(net,...
                        theta, normStat, states, maxIdx); 
                    [~, ~, ma, Sa]   = tagi.extractStates(states);
                    zl(idxBatch, :)  = gather(reshape(ma{end}, ...
                        [net.ny, numDataPerBatch])');
                    Szl(idxBatch, :) = gather(reshape(Sa{end}, ...
                        [net.ny, numDataPerBatch])');
                    sv = net.sv;
                end 
                zl(idxBatch, :)  = gather(reshape(ma{end}, ...
                    [net.ny, numDataPerBatch])');
                Szl(idxBatch, :) = gather(reshape(Sa{end}, ...
                    [net.ny, numDataPerBatch])');
            end
        end
        
        % Autoencoder
        function [thetaE, thetaD, normStatE, normStatD] = AE(netE, thetaE,...
                normStatE, statesE, maxIdxE, netD, thetaD, normStatD,...
                statesD, maxIdxD, x, Niter)
            % Initialization
            numObs = size(x, 4);
            numDataPerBatch = netE.batchSize * netE.repBatchSize;
            % Loop
            loop = 0;
            for i = 1:Niter
                loop     = loop + 1;
                idxBatch = randperm(numObs, numDataPerBatch);
                xloop    = dp.dataLoader(x(:,:,:,idxBatch), netE.da,...
                    netE.trainMode);
                xloop    = reshape(xloop, [netE.batchSize * netE.nodes(1),...
                    netE.repBatchSize]);
                yloop    = xloop;
                if netE.gpu
                    yloop = gpuArray(yloop);
                    xloop = yloop;
                end
                % Forward
                statesE                       = tagi.initializeInputs(statesE,...
                    xloop, [], [], [], [], [], [], [], [], netE.xsc);
                [statesE, normStatE, maxIdxE] = tagi.feedForwardPass(netE,...
                    thetaE, normStatE, statesE, maxIdxE);
                [mzE, SzE, maE, SaE, JE,...
                    mdxsE, SdxsE, mxsE, SxsE] = tagi.extractStates(statesE);
                
                statesD                       = tagi.initializeInputs(statesD,...
                    mzE{end}, SzE{end}, maE{end}, SaE{end}, JE{end},...
                    mdxsE{end}, SdxsE{end}, mxsE{end}, SxsE{end}, netE.xsc);
                [statesD, normStatD, maxIdxD] = tagi.feedForwardPass(netD,...
                    thetaD, normStatD, statesD, maxIdxD);
                [~, ~, maD, SaD]              = tagi.extractStates(statesD);
                
                % Backward
                [deltaMD, deltaSD, deltaMxD, deltaSxD,...
                 deltaMz0D, deltaSz0D] = tagi.hiddenStateBackwardPass(netD,...
                 thetaD, normStatD, statesD, yloop, [], [], maxIdxD);
                deltaThetaD = tagi.parameterBackwardPass(netD, thetaD,...
                    normStatD, statesD, deltaMD, deltaSD, deltaMxD, deltaSxD);
                thetaD      = tagi.globalParameterUpdate(thetaD,...
                    deltaThetaD, netD.gpu);
                
                [deltaME, deltaSE,...
                    deltaMxE, deltaSxE] = tagi.hiddenStateBackwardPass(netE,...
                    thetaE, normStatE, statesE, deltaMz0D, deltaSz0D, [], ...
                    maxIdxE);
                deltaThetaE             = tagi.parameterBackwardPass(netE,...
                    thetaE, normStatE, statesE, deltaME, deltaSE, deltaMxE,...
                    deltaSxE);
                thetaE                  = tagi.globalParameterUpdate(thetaE,...
                    deltaThetaE, netE.gpu);
                if netD.learnSv == 1
                    ml = tagi.detachMeanVar(maD{end}, netD.nl, netD.nv2,...
                        netD.batchSize);
                    Z  = gather(reshape(ml, [netD.nl, netD.batchSize])');
                else
                    Z = gather(reshape(maD{end}, [netD.ny, ...
                        netD.batchSize * netD.repBatchSize])');
                end
                if any(isnan(SaD{end}(:,1)))
                    error('Decoder variance is nan')
                end
            end
        end
        function [theta, normStat] = AE_V2(net, theta, normStat, states,...
                maxIdx, x, y)
            % Initialization           
            numObs = size(x, 4);
            rB     = net.repBatchSize;
            B      = net.batchSize;
            sv     = net.sv;
            nyc    = size(y, 2);
            numDataPerBatch = rB * B;
            if net.learnSv == 1
                ny = net.ny - net.nv2;
            else
                ny = net.ny;
            end
            if net.errorRateEval == 1
                Pn = zeros(numObs, net.numClasses, net.dtype);
                er = zeros(numObs, 1, net.dtype);
                [classObs, classIdx] = dp.class_encoding(net.numClasses);
                addIdx   = reshape(repmat(colon(0, nyc, (numDataPerBatch - 1) * nyc),...
                    [net.numClasses, 1]), [net.numClasses * numDataPerBatch, 1]);
                classObs = repmat(classObs, [numDataPerBatch, 1]);
                classIdx = repmat(classIdx, [numDataPerBatch, 1]) +...
                    cast(addIdx, class(classIdx));
            else
                er = nan;
                Pn = nan;
            end
            % Loop
            loop     = 0;
            time_tot = 0;
            for i = 1:numDataPerBatch:numObs
                timeval  = tic;
                loop     = loop + 1;
                idxBatch = i : i + numDataPerBatch - 1;
                xloop    = dp.dataLoader(x(:,:,:,idxBatch), net.da,...
                    net.batchSize, net.repBatchSize, net.trainMode);
                xloop    = reshape(xloop, [net.batchSize * net.nodes(1),...
                    net.repBatchSize]);
                states   = tagi.initializeInputs(states, xloop, [], [], [],...
                    [], [], [], [], [], net.xsc);
                
                % Training
                [yloop, Syloop,...
                    udIdx] = network.labelLatentVarDist(y(idxBatch, :), sv,...
                    net.encoderIdx(idxBatch, :), net.numLatentVar, B, rB,...
                    net.dtype);              
                [states, normStat, maxIdx] = tagi.feedForwardPass(net,...
                    theta, normStat, states, maxIdx);
                [deltaM, deltaS,deltaMx,...
                    deltaSx] = tagi.hiddenStateBackwardPass(net, theta,...
                    normStat, states, yloop, Syloop, udIdx, maxIdx);
                dtheta = tagi.parameterBackwardPass(net, theta, normStat,...
                    states, deltaM, deltaS, deltaMx, deltaSx);
                theta  = tagi.globalParameterUpdate(theta, dtheta, net.gpu);

                [~, ~, ma, Sa] = tagi.extractStates(states);
                ml = reshape(ma{end}, [ny, numDataPerBatch])';
                Sl = reshape(Sa{end}, [ny, numDataPerBatch])';
                ml = reshape(ml(:, 1:nyc)', [numDataPerBatch * nyc, 1]);
                Sl = reshape(Sl(:, 1:nyc)' + sv.^2, [numDataPerBatch * nyc, 1]);
                
                time_loop = toc(timeval);
                time_tot  = time_tot + time_loop;
                time_rem  = double(time_tot) / (double(idxBatch(end))) ...
                    * (numObs-double(idxBatch(end))) / 60;
                if net.errorRateEval == 1
                    P = dp.obs2class(ml, Sl, classObs, classIdx);
                    P = reshape(P, [net.numClasses, numDataPerBatch])';
                    P = gather(P);
                    Pn(idxBatch, :) = P;
                    er(idxBatch, :) = mt.errorRate(net.labels(idxBatch, :)', P');   
%                     % Display error rate  
                    if mod(idxBatch(end), net.obsShow) == 0 ...
                            && i > 1 ...
                            && net.trainMode == 1 ...
                            && net.displayMode == 1
                        disp(['     Error Rate : ' ...
                            sprintf('%0.2f', 100 * mean(er(max(1,i - 200) : i))) '%']);
                        disp(['     Time left  : ' ...
                            sprintf('%0.2f',time_rem) ' mins'])
                    end
                else
                    if mod(idxBatch(end), net.obsShow) == 0 ...
                            && i > 1 && net.trainMode == 1 ...
                            && net.displayMode == 1
                        disp(['     Time left : ' ...
                            sprintf('%0.2f',time_rem) ' mins'])
                    end
                end                               
            end
        end
        function [mxpost, Sxpost] = generateX(net, theta, normStat, states,...
                maxIdx, mxprior, Sxprior, y, udIdx)                          
            % Generating
            states   = tagi.initializeInputs(states, mxprior, Sxprior, [],...
                [], [], [], [], [], [], net.xsc);
            [states, normStat, maxIdx] = tagi.feedForwardPass(net, theta,...
                normStat, states, maxIdx);
            [~, ~, ~, ~, ...
                deltaMz0, deltaSz0] = tagi.hiddenStateBackwardPass(net, ...
                theta, normStat, states, y, [], udIdx, maxIdx);
            
            % Posterior
            mxpost = mxprior + deltaMz0;
            Sxpost = Sxprior + deltaSz0;
        end         
        function [img] = genAE_V1(netE, thetaE, normStatE, statesE, maxIdxE,...
                netD, thetaD, normStatD, statesD, maxIdxD, x, numSamples, latentVar)
            % Initialization
            if netE.gpu
                zeroPad = zeros(1, 1, netE.dtype, 'gpuArray');
            else
                zeroPad = zeros(1, 1, netE.dtype);
            end
            numObs = size(x, 4);
            numDataPerBatch = netE.batchSize * netE.repBatchSize;
            mlv = zeros(netD.nx, numObs, 'like', zeroPad);
            Slv = zeros(netD.nx, numObs, 'like', zeroPad);
            img = zeros(numObs * numSamples, netE.nx);
            
            % Loop
            loop = 0;
            for i = 1 : numDataPerBatch : numObs
                loop     = loop + 1;
                idxBatch = i:i+numDataPerBatch-1;
                xloop    = dp.dataLoader(x(:,:,:,idxBatch), netE.da, ...
                    netE.trainMode);
                xloop    = reshape(xloop, [netE.batchSize*netE.nodes(1), ...
                    netE.repBatchSize]);                                          
                if netE.gpu
                    xloop = gpuArray(xloop);
                end               
                % Forward pass
                statesE                       = tagi.initializeInputs(statesE,...
                    xloop, [], [], [], [], [], [], [], [], netE.xsc);  
                [statesE, normStatE, maxIdxE] = tagi.feedForwardPass(netE,...
                    thetaE, normStatE, statesE, maxIdxE);
                [~, ~, maE, SaE] = tagi.extractStates(statesE);  
                mlv(:, idxBatch) = reshape(maE{end}, [netE.ny, netE.batchSize * netE.repBatchSize]);
                Slv(:, idxBatch) = reshape(SaE{end}, [netE.ny, netE.batchSize * netE.repBatchSize]);
            end
            
            % Generate data
            lvSamples = reshape(repmat(mlv, [numSamples, 1]), ...
                [netD.nx, numSamples * numObs]);
            loop_lv = 0;
            for s = 1 : numSamples : numObs * numSamples
                loop_lv = loop_lv + 1;
                idxBatch = s : s + numSamples - 1;
                lvSamples(latentVar, idxBatch) = sort(normrnd(mlv(latentVar,...
                    loop_lv), 2*sqrt(Slv(latentVar, loop_lv)),...
                    [1, numSamples]));
            end
                        
            for i = 1 : numDataPerBatch : numObs * numSamples
                loop     = loop + 1;
                idxBatch = i : i + numDataPerBatch - 1;
                lvloop   = lvSamples(:, idxBatch);
                lvloop   = reshape(lvloop, ...
                    [netD.nx * netD.batchSize * netD.repBatchSize, 1]);                                                                  
                statesD                       = tagi.initializeInputs(statesD,...
                    lvloop, [], [], [], [], [], [], [], [], netE.xsc); 
                [statesD, normStatD, maxIdxD] = tagi.feedForwardPass(netD,...
                    thetaD, normStatD, statesD, maxIdxD);
                [~, ~, maD]              = tagi.extractStates(statesD);                
                
                img(idxBatch, :) = gather(reshape(maD{end}, ...
                    [netD.ny, netD.batchSize*netD.repBatchSize])');                          
            end
            
        end
        
        % GAN
        function [thetaD, thetaG, normStatD, normStatG] = GAN(netD, thetaD,...
                normStatD, statesD, maxIdxD, netG, thetaG, normStatG, ...
                statesG, maxIdxG, xD)
            % Initialization
            numObs = size(xD, 4);
            numDataPerBatch = netD.batchSize * netD.repBatchSize;
            if netD.gpu
                zeroPad = zeros(1, 1, netD.dtype, 'gpuArray');
            else
                zeroPad = zeros(1, 1, netD.dtype);
            end
            % Loop
            loop     = 0;
            for i = 1 : numDataPerBatch : numObs
                loop     = loop + 1;
                idxBatch = i : i + numDataPerBatch - 1;
                xloopG   = reshape(randn(netG.nx, ...
                    netG.batchSize * netG.repBatchSize, 'like', zeroPad),...
                    [netG.nx * netG.batchSize, netG.repBatchSize]);
                xloop    = dp.dataLoader(xD(:,:,:,idxBatch), netD.da,...
                    netD.batchSize, netD.repBatchSize, netD.trainMode); 
                xloop    = reshape(xloop, [netD.batchSize * netD.nodes(1),...
                    netD.repBatchSize]);  
                
                yfake    = ones(netD.batchSize, netD.repBatchSize, ...
                    'like', zeroPad);
                yreal    = -ones(netD.batchSize, netD.repBatchSize, ...
                    'like', zeroPad);
                
                % Update dicriminator (netD)
                    % Real example
                statesD                       = tagi.initializeInputs(statesD,...
                    xloop, [], [], [], [], [], [], [], [], netD.xsc);  
                [statesD, normStatD, maxIdxD] = tagi.feedForwardPass(netD,...
                    thetaD, normStatD, statesD, maxIdxD); 
                
                [deltaMD, deltaSD,...
                    deltaMxD, deltaSxD] = tagi.hiddenStateBackwardPass(netD,...
                    thetaD, normStatD, statesD, yreal, [], [], maxIdxD);
                deltaThetaD             = tagi.parameterBackwardPass(netD,...
                    thetaD, normStatD, statesD, deltaMD, deltaSD, deltaMxD,...
                    deltaSxD);
                thetaD                  = tagi.globalParameterUpdate(thetaD,...
                    deltaThetaD);  
                    % Fake examples
                statesG                       = tagi.initializeInputs(statesG,...
                    xloopG, [], [], [], [], [], [], [], [], netG.xsc);  
                [statesG, normStatG, maxIdxG] = tagi.feedForwardPass(netG,...
                    thetaG, normStatG, statesG, maxIdxG);
                [mzG, SzG, maG, SaG, JG,...
                    mdxsG, SdxsG, mxsG, SxsG] = tagi.extractStates(statesG);
                statesD                       = tagi.initializeInputs(statesD,...
                    mzG{end}, SzG{end}, maG{end}, SaG{end}, JG{end}, mdxsG{end},...
                    SdxsG{end}, mxsG{end}, SxsG{end}, netD.xsc); 
                [statesD, normStatD, maxIdxD] = tagi.feedForwardPass(netD,...
                    thetaD, normStatD, statesD, maxIdxD);
                
                [deltaMD, deltaSD,...
                    deltaMxD, deltaSxD] = tagi.hiddenStateBackwardPass(netD,...
                    thetaD, normStatD, statesD, yfake, [], [], maxIdxD);
                deltaThetaD             = tagi.parameterBackwardPass(netD,...
                    thetaD, normStatD, statesD, deltaMD, deltaSD, deltaMxD,...
                    deltaSxD);
                thetaD                  = tagi.globalParameterUpdate(thetaD,...
                    deltaThetaD); 
                
                % Update generator (netG)
                statesD                       = tagi.initializeInputs(statesD,...
                    mzG{end}, SzG{end}, maG{end}, SaG{end}, JG{end}, mdxsG{end},...
                    SdxsG{end}, mxsG{end}, SxsG{end}, netD.xsc); 
                [statesD, normStatD, maxIdxD] = tagi.feedForwardPass(netD,...
                    thetaD, normStatD, statesD, maxIdxD); 
                [~, ~, ~, ~,...
                    deltaMz0D, deltaSz0D]     = tagi.hiddenStateBackwardPass(netD,...
                    thetaD, normStatD, statesD, yreal, [], [], maxIdxD);
                
                [deltaMG, deltaSG,...
                    deltaMxG, deltaSxG] = tagi.hiddenStateBackwardPass(netG,...
                    thetaG, normStatG, statesG, deltaMz0D, deltaSz0D, [],...
                    maxIdxG);
                deltaThetaG             = tagi.parameterBackwardPass(netG,...
                    thetaG, normStatG, statesG, deltaMG, deltaSG, deltaMxG,...
                    deltaSxG);
                thetaG                  = tagi.globalParameterUpdate(thetaG,...
                    deltaThetaG); 
                
                Z  = gather(reshape(maG{end}, [netG.ny, length(idxBatch)])');
                if any(isnan(SzG{end}(:, 1)))
                    error('Generator variance is nan')
                end                          
            end
        end        
         
        % infoGAN
        function [thetaD, thetaG, thetaQ, thetaP, normStatD, normStatG,...
                normStatQ, normStatP] = infoGAN(netD, thetaD, normStatD,...
                statesD, maxIdxD, netG, thetaG, normStatG, statesG, maxIdxG,...
                netQ, thetaQ, normStatQ, statesQ, maxIdxQ, netP, thetaP,...
                normStatP, statesP, maxIdxP, xD, Niter)
            % Initialization
            numObs = size(xD, 4);
            numDataPerBatch = netD.batchSize * netD.repBatchSize;
            % Loop
            loop     = 0;
            for i = 1 : Niter
                loop     = loop + 1;
                idxBatch = randperm(numObs, numDataPerBatch);
                
                % Form a vector including random noise + categorical
                % variables + continuous variables
                [xcc, udIdxQ, ...
                    xloopG] = network.generateLatentVar(netQ.numClasses,...
                    netQ.numCatVar, netQ.numContVar, numDataPerBatch, netG.nx, netQ.dtype, netQ.gpu); 
                xcc    = reshape(xcc, [numel(xcc) / (netQ.repBatchSize),...
                    netQ.repBatchSize]);
                udIdxQ = reshape(udIdxQ, [numel(udIdxQ) / (netQ.repBatchSize),...
                    netQ.repBatchSize]);
                xloopG = reshape(xloopG, [numel(xloopG) / (netQ.repBatchSize),...
                    netQ.repBatchSize]);
                
                % Real image pixels
                xloop  = dp.dataLoader(xD(:,:,:,idxBatch), netD.da, ...
                    netD.batchSize, netD.repBatchSize, netD.trainMode); 
                xloop  = reshape(xloop, [netD.batchSize*netD.nodes(1), ...
                    netD.repBatchSize]);  
                if netD.gpu||netG.gpu||netQ.gpu||netP.gpu
                    xloop  = gpuArray(xloop);
                end
                yfake = ones(netD.batchSize, netD.repBatchSize, 'like', xloop);
                yreal = -ones(netD.batchSize, netD.repBatchSize, 'like', xloop);
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %Update discriminator (netD, netP, and netQ)
                statesD                   = tagi.initializeInputs(statesD,...
                    xloop, [], [], [], [], [], [], [], [], netD.xsc);
                [thetaD, thetaP,...
                    normStatD, normStatP] = network.updateDPinfoGAN(netD,...
                    thetaD, normStatD, statesD, maxIdxD, ...
                netP, thetaP, normStatP, statesP, maxIdxP, yreal);
            
                % Generate fake examples using netG
                statesG                       = tagi.initializeInputs(statesG,...
                    xloopG, [], [], [], [], [], [], [], [], netG.xsc);  
                [statesG, normStatG, maxIdxG] = tagi.feedForwardPass(netG,...
                    thetaG, normStatG, statesG, maxIdxG);
                [mzG, SzG, maG, SaG, JG,...
                    mdxsG, SdxsG, mxsG, SxsG] = tagi.extractStates(statesG);
                
                % Feed fake examples to netD
                statesD                              = tagi.initializeInputs(statesD,...
                    mzG{end}, SzG{end}, maG{end}, SaG{end}, JG{end},...
                    mdxsG{end}, SdxsG{end}, mxsG{end}, SxsG{end}, netD.xsc);  
                [thetaD, thetaP, thetaQ,...
                    normStatD, normStatP, normStatQ] = network.updateDPQacGAN(netD,...
                    thetaD, normStatD, statesD, maxIdxD, netQ, thetaQ, ...
                    normStatQ, statesQ, maxIdxQ, netP, thetaP, normStatP, ...
                    statesP, maxIdxP, yfake, xcc, udIdxQ);         
                                   
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %%%% Update generator (netG)
                % Feed fake examples to netD
                statesD                       = tagi.initializeInputs(statesD,...
                    mzG{end}, SzG{end}, maG{end}, SaG{end}, JG{end},...
                    mdxsG{end}, SdxsG{end}, mxsG{end}, SxsG{end}, netD.xsc);  
                [statesD, normStatD, maxIdxD] = tagi.feedForwardPass(netD,...
                    thetaD, normStatD, statesD, maxIdxD);               
                [mzD, SzD, maD, SaD, JD,...
                    mdxsD, SdxsD, mxsD, SxsD] = tagi.extractStates(statesD);
                
                % Feed netD's outputs to netP
                statesP                       = tagi.initializeInputs(statesP,...
                    mzD{end}, SzD{end}, maD{end}, SaD{end}, JD{end},...
                    mdxsD{end}, SdxsD{end}, mxsD{end}, SxsD{end}, netP.xsc);  
                [statesP, normStatP, maxIdxP] = tagi.feedForwardPass(netP,...
                    thetaP, normStatP, statesP, maxIdxP);
                
                % Feed netD's output to the 2nd head (netQ) that infers
                % latent variables
                statesQ                       = tagi.initializeInputs(statesQ,...
                    mzD{end}, SzD{end}, maD{end}, SaD{end}, JD{end},...
                    mdxsD{end}, SdxsD{end}, mxsD{end}, SxsD{end}, netQ.xsc);  
                [statesQ, normStatQ, maxIdxQ] = tagi.feedForwardPass(netQ,...
                    thetaQ, normStatQ, statesQ, maxIdxQ);       
                
                % Update hidden states for netP 
                [~, ~, ~, ~,...
                    deltaMz0P, deltaSz0P] = tagi.hiddenStateBackwardPass(netP,...
                    thetaP, normStatP, statesP, yreal, [], [], maxIdxP);
                % Update parameters & hidden states for netQ
                [~, ~, ~, ~, ...
                    deltaMz0Q, deltaSz0Q] = tagi.hiddenStateBackwardPass(netQ,...
                    thetaQ, normStatQ, statesQ, xcc, [], udIdxQ, maxIdxQ);
                
                deltaMzLD = deltaMz0P + deltaMz0Q;
                deltaSzLD = deltaSz0P + deltaSz0Q;
                [~, ~, ~, ~,...
                    deltaMz0D, deltaSz0D] = tagi.hiddenStateBackwardPass(netD,...
                    thetaD, normStatD, statesD, deltaMzLD, deltaSzLD, [],...
                    maxIdxD);   
                
                % Update parameters & hidden states for netG
                [deltaMG, deltaSG,...
                    deltaMxG, deltaSxG] = tagi.hiddenStateBackwardPass(netG,...
                    thetaG, normStatG, statesG, deltaMz0D, deltaSz0D, [],...
                    maxIdxG);
                deltaThetaG             = tagi.parameterBackwardPass(netG,...
                    thetaG, normStatG, statesG, deltaMG, deltaSG, deltaMxG,...
                    deltaSxG);
                thetaG                  = tagi.globalParameterUpdate(thetaG,...
                    deltaThetaG, netG.gpu); 

                Z  = gather(reshape(maG{end}, ...
                    [netG.ny, netG.batchSize * netG.repBatchSize])');               
                if any(isnan(SzG{end}(:, 1)))
                    error('Generator variance is nan')
                end
                
                % Set momentum for average running after 1st obs                   
                netD.normMomentum = netD.normMomentumRef;
                netP.normMomentum = netP.normMomentumRef;
                netQ.normMomentum = netQ.normMomentumRef;
                netG.normMomentum = netG.normMomentumRef;
            end
        end
        
        % ACGAN
        function [thetaD, thetaG, thetaQ, thetaP, normStatD, normStatG,...
                normStatQ, normStatP] = ACGAN(netD, thetaD, normStatD,...
                statesD, maxIdxD, netG, thetaG, normStatG, statesG, maxIdxG,...
                netQ, thetaQ, normStatQ, statesQ, maxIdxQ, netP, thetaP,...
                normStatP, statesP, maxIdxP, x, y, udIdx)
            % Initialization
            numObs = size(x, 4);
            numDataPerBatch = netD.batchSize * netD.repBatchSize;
            % Loop
            loop = 0;
            if netD.gpu || netG.gpu || netQ.gpu || netP.gpu
                yPfake = ones(netP.batchSize, netP.repBatchSize,...
                    netP.dtype, 'gpuArray');
                yPreal = -ones(netP.batchSize, netP.repBatchSize,...
                    netP.dtype, 'gpuArray');
            else
                yPfake = ones(netP.batchSize, netP.repBatchSize, 'like', x);
                yPreal = -ones(netP.batchSize, netP.repBatchSize, 'like', x);
            end
            for i = 1 : numDataPerBatch : numObs
                loop     = loop + 1;
                idxBatch = i : i + numDataPerBatch - 1;
                % Noise and fake labels
                [yQfake, udIdxQfake] = network.generateLabels(netQ.numClasses,...
                    netQ.numCatVar, numDataPerBatch, netQ.dtype);               
                udIdxQfake = dp.selectIndices(udIdxQfake, numDataPerBatch,...
                    netQ.ny, netQ.dtype);
                xG         = [randn(numDataPerBatch, netG.nx - netQ.ny),...
                    yQfake];
                xG         = reshape(xG', [netG.nx * netQ.batchSize,...
                    netQ.repBatchSize]);
                yQfake     = reshape(yQfake', [netQ.batchSize * netQ.ny,...
                    netQ.repBatchSize]);
                % Real images and labels
                xD         = dp.dataLoader(x(:,:,:,idxBatch), netD.da,...
                    netD.batchSize, netD.repBatchSize, netD.trainMode); 
                xD         = reshape(xD, [netD.batchSize * netD.nodes(1),...
                    netD.repBatchSize]);  
                yQreal     = reshape(y(idxBatch, :)', ...
                    [netQ.batchSize * netQ.ny, netQ.repBatchSize]);
                udIdxQreal = dp.selectIndices(udIdx(idxBatch, :), ...
                    numDataPerBatch, netQ.ny, netQ.dtype);
                if netD.gpu || netG.gpu || netQ.gpu || netP.gpu
                    xG     = gpuArray(xG);
                    xD     = gpuArray(xD);
                    yQfake = gpuArray(yQfake);
                    yQreal = gpuArray(yQreal);
                    udIdxQfake = gpuArray(udIdxQfake);
                    udIdxQreal = gpuArray(udIdxQreal);  
                end  
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %Update discriminator (netD, netP, and netQ)
                % Feed real examples to netD
                statesD = tagi.initializeInputs(statesD, xD, [], [], [], [],...
                    [], [], [], [], netD.xsc); 
                [thetaD, thetaP, thetaQ, normStatD,...
                    normStatP, normStatQ] = network.updateDPQacGAN(netD,...
                    thetaD, normStatD, statesD, maxIdxD, ...
                netQ, thetaQ, normStatQ, statesQ, maxIdxQ, netP, thetaP,...
                normStatP, statesP, maxIdxP, yPreal, yQreal, udIdxQreal);               
                                
                % Generate fake examples using netG
                statesG                       = tagi.initializeInputs(statesG,...
                    xG, [], [], [], [], [], [], [], [], netG.xsc);  
                [statesG, normStatG, maxIdxG] = tagi.feedForwardPass(netG,...
                    thetaG, normStatG, statesG, maxIdxG);
                [mzG, SzG, maG, SaG, JG,...
                    mdxsG, SdxsG, mxsG, SxsG] = tagi.extractStates(statesG);
                
                % Feed fake examples to netD
                statesD = tagi.initializeInputs(statesD, mzG{end}, SzG{end},...
                    maG{end}, SaG{end}, JG{end}, mdxsG{end}, SdxsG{end},...
                    mxsG{end}, SxsG{end}, netD.xsc);  
                [thetaD, thetaP, thetaQ, normStatD,...
                    normStatP, normStatQ] = network.updateDPQacGAN(netD, ...
                    thetaD, normStatD, statesD, maxIdxD, ...
                netQ, thetaQ, normStatQ, statesQ, maxIdxQ, netP, thetaP,...
                normStatP, statesP, maxIdxP, yPfake, yQfake, udIdxQfake); 
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %%%% Update generator (netG)
                % Feed fake examples to netD
                statesD                       = tagi.initializeInputs(statesD,...
                    mzG{end}, SzG{end}, maG{end}, SaG{end}, JG{end},...
                    mdxsG{end}, SdxsG{end}, mxsG{end}, SxsG{end}, netD.xsc);  
                [statesD, normStatD, maxIdxD] = tagi.feedForwardPass(netD,...
                    thetaD, normStatD, statesD, maxIdxD);               
                [mzD, SzD, maD, SaD, JD,...
                    mdxsD, SdxsD, mxsD, SxsD] = tagi.extractStates(statesD);
                
                % Feed netD's outputs to netP
                statesP                       = tagi.initializeInputs(statesP,...
                    mzD{end}, SzD{end}, maD{end}, SaD{end}, JD{end}, mdxsD{end}, SdxsD{end}, mxsD{end}, SxsD{end}, netP.xsc);  
                [statesP, normStatP, maxIdxP] = tagi.feedForwardPass(netP,...
                    thetaP, normStatP, statesP, maxIdxP);
                
                % Feed netD's outputs to netQ
                statesQ                       = tagi.initializeInputs(statesQ,...
                    mzD{end}, SzD{end}, maD{end}, SaD{end}, JD{end}, ...
                    mdxsD{end}, SdxsD{end}, mxsD{end}, SxsD{end}, netQ.xsc);
                [statesQ, normStatQ, maxIdxQ] = tagi.feedForwardPass(netQ, ...
                    thetaQ, normStatQ, statesQ, maxIdxQ);
                
                % Update hidden states for netQ, netP, & netD
                [~, ~, ~, ~, ...
                    deltaMz0P, deltaSz0P] = tagi.hiddenStateBackwardPass(netP,...
                    thetaP, normStatP, statesP, yPreal, [], [], maxIdxP);
                [~, ~, ~, ~, ...
                    deltaMz0Q, deltaSz0Q] = tagi.hiddenStateBackwardPass(netQ,...
                    thetaQ, normStatQ, statesQ, yQfake, [], udIdxQfake, maxIdxQ);
                deltaMzLD = deltaMz0P + deltaMz0Q;
                deltaSzLD = deltaSz0P + deltaSz0Q;
                [~, ~, ~, ~, ...
                    deltaMz0D, deltaSz0D] = tagi.hiddenStateBackwardPass(netD,...
                    thetaD, normStatD, statesD, deltaMzLD , deltaSzLD , [], maxIdxD);
                
                % Update parameters & hidden states for netG
                [deltaMG, deltaSG,...
                    deltaMxG, deltaSxG] = tagi.hiddenStateBackwardPass(netG,...
                    thetaG, normStatG, statesG, deltaMz0D, deltaSz0D, [],...
                    maxIdxG);
                deltaThetaG             = tagi.parameterBackwardPass(netG,...
                    thetaG, normStatG, statesG, deltaMG, deltaSG, deltaMxG,...
                    deltaSxG);
                thetaG                  = tagi.globalParameterUpdate(thetaG,...
                    deltaThetaG); 
                
                Z = gather(reshape(maG{end}, [netG.ny, length(idxBatch)])');                
                if any(isnan(SzG{end}(:, 1)))||any(SzG{end}(:, 1)<0)
                    error('Generator variance is nan');
                end                                          
            end
        end     
        
        % Sharing funtions
        function [thetaD, thetaQ, thetaQc, thetaP, normStatD, normStatQ,...
                normStatQc, normStatP] = updateDPQinfoGAN(netD, thetaD,...
                normStatD, statesD, maxIdxD, netQ, thetaQ, normStatQ, ...
                statesQ, maxIdxQ, netQc, thetaQc, normStatQc, statesQc, ...
                maxIdxQc, netP, thetaP, normStatP, statesP, maxIdxP, ...
                yQ, yQc, yP, udIdxQ)
            % Feed real examples to netD
            [statesD, normStatD, maxIdxD] = tagi.feedForwardPass(netD, ...
                thetaD, normStatD, statesD, maxIdxD);
            [mzD, SzD, maD, SaD, JD,...
                mdxsD, SdxsD, mxsD, SxsD] = tagi.extractStates(statesD);
            
            % Feed netD's output to the 1st head (netP) that
            % discriminates fake/real images
            statesP                       = tagi.initializeInputs(statesP,...
                mzD{end}, SzD{end}, maD{end}, SaD{end}, JD{end}, mdxsD{end},...
                SdxsD{end}, mxsD{end}, SxsD{end}, netP.xsc);
            [statesP, normStatP, maxIdxP] = tagi.feedForwardPass(netP, ...
                thetaP, normStatP, statesP, maxIdxP);
            
            % Feed netD's output to the 2nd head (netQ) that infers
            % discrete slatent variables            
            statesQ                       = tagi.initializeInputs(statesQ, ...
                mzD{end}, SzD{end}, maD{end}, SaD{end}, JD{end}, mdxsD{end}, ...
                SdxsD{end}, mxsD{end}, SxsD{end}, netQ.xsc);
            [statesQ, normStatQ, maxIdxQ] = tagi.feedForwardPass(netQ, ...
                thetaQ, normStatQ, statesQ, maxIdxQ);
            
            % Feed netD's output to the 3rd head (netQc) that infers
            % continuous latent variables
            statesQc                         = tagi.initializeInputs(statesQc,...
                mzD{end}, SzD{end}, maD{end}, SaD{end}, JD{end}, mdxsD{end},...
                SdxsD{end}, mxsD{end}, SxsD{end}, netQc.xsc);
            [statesQc, normStatQc, maxIdxQc] = tagi.feedForwardPass(netQc,...
                thetaQc, normStatQc, statesQc, maxIdxQc);
            
            % Update parameters & hidden states for netP
            [deltaMP, deltaSP, deltaMxP, deltaSxP, deltaMz0P,...
                deltaSz0P] = tagi.hiddenStateBackwardPass(netP, thetaP, ...
                normStatP, statesP, yP, [], [], maxIdxP);
            deltaThetaP = tagi.parameterBackwardPass(netP, thetaP,...
                normStatP, statesP, deltaMP, deltaSP, deltaMxP, deltaSxP);           
            thetaP  = tagi.globalParameterUpdate(thetaP, deltaThetaP);
            
            % Update parameters & hidden states for netQ
            [deltaMQ, deltaSQ, deltaMxQ, deltaSxQ, deltaMz0Q,...
                deltaSz0Q] = tagi.hiddenStateBackwardPass(netQ, thetaQ,...
                normStatQ, statesQ, yQ, [], udIdxQ, maxIdxQ);
            deltaThetaQ = tagi.parameterBackwardPass(netQ, thetaQ,...
                normStatQ, statesQ, deltaMQ, deltaSQ, deltaMxQ, deltaSxQ);
            thetaQ      = tagi.globalParameterUpdate(thetaQ, deltaThetaQ);
            
            % Update parameters & hidden states for netQc
            [deltaMQc, deltaSQc, deltaMxQc, deltaSxQc, deltaMz0Qc,...
                deltaSz0Qc] = tagi.hiddenStateBackwardPass(netQc, thetaQc,...
                normStatQc, statesQc, yQc, [], [], maxIdxQc);
            deltaThetaQc = tagi.parameterBackwardPass(netQc, thetaQc,...
                normStatQc, statesQc, deltaMQc, deltaSQc, deltaMxQc, deltaSxQc);
            thetaQc = tagi.globalParameterUpdate(thetaQc, deltaThetaQc);
            
            % Update parameters & hidden states for Dnent from netQ and netP
            deltaMzLD = deltaMz0P + deltaMz0Q + deltaMz0Qc;
            deltaSzLD = deltaSz0P + deltaSz0Q + deltaSz0Qc;
            [deltaMD, deltaSD,...
                deltaMxD, deltaSxD] = tagi.hiddenStateBackwardPass(netD,...
                thetaD, normStatD, statesD,deltaMzLD, deltaSzLD, [], maxIdxD);
            deltaThetaD = tagi.parameterBackwardPass(netD, thetaD,...
                normStatD, statesD, deltaMD, deltaSD, deltaMxD, deltaSxD);
            thetaD      = tagi.globalParameterUpdate(thetaD, deltaThetaD);
        end 
        function [thetaD, thetaP, thetaQ, normStatD, normStatP,...
                normStatQ] = updateDPQacGAN(netD, thetaD, normStatD,...
                statesD, maxIdxD, netQ, thetaQ, normStatQ, statesQ, maxIdxQ,...
                netP, thetaP, normStatP, statesP, maxIdxP, yP, yQ, udIdxQ)
            % Feed real examples to netD
            [statesD, normStatD, maxIdxD] = tagi.feedForwardPass(netD,...
                thetaD, normStatD, statesD, maxIdxD);
            [mzD, SzD, maD, SaD, JD,...
                mdxsD, SdxsD, mxsD, SxsD] = tagi.extractStates(statesD);
            
            % Feed netD's output to the 1st head (netP) that
            % discriminates fake/real images
            statesP                       = tagi.initializeInputs(statesP,...
                mzD{end}, SzD{end}, maD{end}, SaD{end}, JD{end}, mdxsD{end},...
                SdxsD{end}, mxsD{end}, SxsD{end}, netP.xsc);
            [statesP, normStatP, maxIdxP] = tagi.feedForwardPass(netP,...
                thetaP, normStatP, statesP, maxIdxP);
            
            % Feed netD's output to the 2nd head (netQ) that infers
            % latent variables
            statesQ                       = tagi.initializeInputs(statesQ,...
                mzD{end}, SzD{end}, maD{end}, SaD{end}, JD{end}, mdxsD{end},...
                SdxsD{end}, mxsD{end}, SxsD{end}, netQ.xsc);
            [statesQ, normStatQ, maxIdxQ] = tagi.feedForwardPass(netQ,...
                thetaQ, normStatQ, statesQ, maxIdxQ);
            
            % Update parameters & hidden states for netP
            [deltaMP, deltaSP, deltaMxP, deltaSxP, deltaMz0P,...
                deltaSz0P] = tagi.hiddenStateBackwardPass(netP, thetaP,...
                normStatP, statesP, yP, [], [], maxIdxP);
            deltaThetaP = tagi.parameterBackwardPass(netP, thetaP, ...
                normStatP, statesP, deltaMP, deltaSP, deltaMxP, deltaSxP);
            thetaP      = tagi.globalParameterUpdate(thetaP, deltaThetaP,...
                netP.gpu);
            
            % Update parameters & hidden states for netQ
            [deltaMQ, deltaSQ, deltaMxQ, deltaSxQ, deltaMz0Q,...
                deltaSz0Q] = tagi.hiddenStateBackwardPass(netQ, thetaQ,...
                normStatQ, statesQ, yQ, [], udIdxQ, maxIdxQ);
            deltaThetaQ = tagi.parameterBackwardPass(netQ, thetaQ, ...
                normStatQ, statesQ, deltaMQ, deltaSQ, deltaMxQ, deltaSxQ);
            thetaQ      = tagi.globalParameterUpdate(thetaQ, deltaThetaQ, ...
                netQ.gpu);
            
            % Update parameters & hidden states for Dnent from netQ and netP
            deltaMzLD = deltaMz0P + deltaMz0Q;
            deltaSzLD = deltaSz0P + deltaSz0Q;
            [deltaMD, deltaSD,...
                deltaMxD, deltaSxD] = tagi.hiddenStateBackwardPass(netD,...
                thetaD, normStatD, statesD,deltaMzLD, deltaSzLD, [], maxIdxD);
            deltaThetaD             = tagi.parameterBackwardPass(netD,...
                thetaD, normStatD, statesD, deltaMD, deltaSD, deltaMxD, deltaSxD);
            thetaD                  = tagi.globalParameterUpdate(thetaD,...
                deltaThetaD, netD.gpu);
        end            
        function [thetaD, thetaP, normStatD,...
                normStatP] = updateDPinfoGAN(netD, thetaD, normStatD,...
                statesD, maxIdxD, netP, thetaP, normStatP, statesP, maxIdxP,...
                yP)            
            [statesD, normStatD, maxIdxD] = tagi.feedForwardPass(netD,...
                thetaD, normStatD, statesD, maxIdxD);
            [mzD, SzD, maD, SaD, JD,...
                mdxsD, SdxsD, mxsD, SxsD] = tagi.extractStates(statesD);
            
            % Feed netD's output to the 1st head (netP) that
            % discriminates fake/real images
            statesP                       = tagi.initializeInputs(statesP,...
                mzD{end}, SzD{end}, maD{end}, SaD{end}, JD{end}, mdxsD{end}, SdxsD{end}, mxsD{end}, SxsD{end}, netP.xsc);
            [statesP, normStatP, maxIdxP] = tagi.feedForwardPass(netP, ...
                thetaP, normStatP, statesP, maxIdxP);
            
            % Update parameters & hidden states for netP
            [deltaMP, deltaSP, deltaMxP, deltaSxP, deltaMz0P,...
                deltaSz0P] = tagi.hiddenStateBackwardPass(netP, thetaP,...
                normStatP, statesP, yP, [], [], maxIdxP);
            deltaThetaP = tagi.parameterBackwardPass(netP, thetaP, ...
                normStatP, statesP, deltaMP, deltaSP, deltaMxP, deltaSxP);
            thetaP      = tagi.globalParameterUpdate(thetaP, deltaThetaP, ...
                netP.gpu);
            
            % Update parameters & hidden states for Dnent from netP
            [deltaMD, deltaSD,...
                deltaMxD, deltaSxD] = tagi.hiddenStateBackwardPass(netD,...
                thetaD, normStatD, statesD, deltaMz0P, deltaSz0P, [],...
                maxIdxD);
            deltaThetaD = tagi.parameterBackwardPass(netD, thetaD,...
                normStatD, statesD, deltaMD, deltaSD, deltaMxD, deltaSxD);
            thetaD      = tagi.globalParameterUpdate(thetaD, deltaThetaD,...
                netD.gpu);
        end
        
        function [y, Sy, udIdx] = labelLatentVarDist(yclass, sv, udIdxClass,...
                numLatentVar, B, rB, dtype)
            nyclass = size(yclass, 2);
            Syclass        = zeros(size(yclass), dtype);            
            mylatentVar    = 0*normrnd(0, 1, rB * B, numLatentVar);
            SylatentVar    = ones(rB * B, numLatentVar, dtype) - sv^2;
            udIdxlatentVar = repmat(colon(nyclass+1, nyclass+numLatentVar),...
                [rB * B, 1]);
            y     = [yclass, mylatentVar];
            Sy    = [Syclass, SylatentVar];
            udIdx = [udIdxClass,  udIdxlatentVar];
            
            y     = reshape(y', [B * (nyclass + numLatentVar), rB]);
            Sy    = reshape(Sy', [B * (nyclass + numLatentVar), rB]);
            udIdx = dp.selectIndices(udIdx, rB * B, nyclass + numLatentVar,...
                dtype);
        end
        function [y, udIdx]     = labelLatentVarSample(numLatentVar, ...
                numClasses, B, rB, dtype)
            labels = randi(numClasses, [B * rB, 1]) - 1; 
            labels = labels * 0 + 2;% label 4
            labels = reshape(labels', [numel(labels), 1]);
            [yclass, udIdxClass] = dp.encoder(labels, numClasses, dtype);           
            nyclass        = size(yclass, 2);       
            
            mylatentVar    = normrnd(0, 1, rB * B, numLatentVar);
            udIdxlatentVar = repmat(colon(nyclass + 1, nyclass+numLatentVar),...
                [rB * B, 1]);
            
            y     = [yclass, mylatentVar];
            udIdx = [udIdxClass,  udIdxlatentVar];
            
            y     = reshape(y', [B * (nyclass + numLatentVar), rB]);
            udIdx = dp.selectIndices(udIdx, rB * B, nyclass + numLatentVar,...
                dtype);
        end
        function [x, idx, xG]   = generateLatentVar(numClasses, numCatVar,...
                numContVar, B, latentDim, dtype, gpu)
            [xcat, idx1] = network.generateLabels(numClasses, numCatVar, B, dtype);
            numH  = cast(size(xcat, 2), 'like', idx1);
            numUd = cast(size(idx1, 2), 'like', idx1);
            xcat  = reshape(xcat', [numH * numCatVar, B])';
            idx1  = reshape(idx1', [size(idx1, 2) * numCatVar, B])';
            addIdx= reshape(repmat(colon(0, numH, (numCatVar - 1) * numH),...
                [numUd, 1]), [1, numCatVar * numUd]);
            idx1  = idx1 + addIdx;
            if numContVar > 0
                xcont = (rand(B, numContVar) * 2 - 1);
                idx2  = repmat(colon(numH * numCatVar + 1, size(xcat, 2) +...
                    numContVar), [B, 1]);
                x     = [xcat, xcont];
                idx   = [idx1, idx2];
            else
                x   = xcat;
                idx = idx1;
            end
            xG    = [randn(B, latentDim - numH * numCatVar - numContVar), x];
            xG    = reshape(xG', [numel(xG), 1]);
            x     = reshape(x', [numel(x), 1]);
            idx   = dp.selectIndices(idx, B, numH * numCatVar + numContVar,...
                dtype);
            x     = x(idx);
            if gpu == 1
                x   = gpuArray(x);
                idx = gpuArray(idx);
                xG  = gpuArray(xG);
            end
        end
        function [x, idx]       = generateLabels(numClasses, numCat, B, ...
                dtype)
            x = randi(numClasses, [B, numCat]) - 1; 
            x = reshape(x', [numel(x), 1]);
            [x, idx] = dp.encoder(x, numClasses, dtype);
        end       
        function [x, Sx, y, updateIdx] = generateRealSamples(x, Sx, y, ...
                updateIdx, numSamples)
            idx = randperm(size(x, 4), numSamples)';
            x = x(:, :, :, idx);
            if ~isempty(Sx)
                Sx = Sx(idx);
            end
            y = y(idx, :);
            updateIdx = updateIdx(idx, :);
        end  
        
        % Plot
        function [x, idx, xG] = generateLatentVar_plot(numClasses, ...
                numCatVar, numContVar, class, B, latentDim, dtype, gpu)
            [xcat, idx1] = network.generateLabels_plot(numClasses, numCatVar, class, B, dtype);
            numH  = cast(size(xcat, 2), 'like', idx1);
            numUd = cast(size(idx1, 2), 'like', idx1);
            xcat  = reshape(xcat', [numH * numCatVar, B])';
            idx1  = reshape(idx1', [size(idx1, 2) * numCatVar, B])';
            addIdx= reshape(repmat(colon(0, numH, (numCatVar - 1) * numH),...
                [numUd, 1]), [1, numCatVar * numUd]);
            idx1  = idx1 + addIdx;
            if numContVar > 0
                xcont = (0 * rand(B, numContVar) * 2 - 1);
%                 xcont(:,2)=repmat(linspace(-2,2,B)', [1, 1]);
                xcont(:, 1)=repmat(linspace(-2, 2, B)', [1, 1]);
                idx2  = repmat(colon(numH * numCatVar + 1, size(xcat, 2) +...
                    numContVar), [B, 1]);
                x     = [xcat, xcont];
                idx   = [idx1, idx2];
            else
                x   = xcat;
                idx = idx1;
            end
            xG    = [0 * randn(B, latentDim - numH * numCatVar - numContVar),...
                x];
            xG    = reshape(xG', [numel(xG), 1]);
            x     = reshape(x', [numel(x), 1]);
            idx   = dp.selectIndices(idx, B, numH * numCatVar + numContVar,...
                dtype);
            if gpu == 1
                x   = gpuArray(x);
                idx = gpuArray(idx);
                xG  = gpuArray(xG);
            end
        end
        function [x, idx, xG] = generateLatentVarCelebA_plot(numClasses,...
                numCatVar, numContVar, B, latentDim, dtype, gpu)
            [xcat, idx1] = network.generateLabelsCelebA_plot(numClasses,...
                numCatVar, B, dtype);
            numH  = cast(size(xcat, 2), 'like', idx1);
            numUd = cast(size(idx1, 2), 'like', idx1);
            xcat  = reshape(xcat', [numH * numCatVar, B])';
            idx1  = reshape(idx1', [size(idx1, 2) * numCatVar, B])';
            addIdx= reshape(repmat(colon(0, numH, (numCatVar - 1) * numH),...
                [numUd, 1]), [1, numCatVar * numUd]);
            idx1  = idx1 + addIdx;
            if numContVar > 0
                xcont = (0  *rand(B, numContVar) * 2 - 1);
%                 xcont(:,2)=repmat(linspace(-2,2,B)', [1, 1]);
%                 xcont(:,1)=repmat(linspace(-,3,B)', [1, 1]);
                idx2  = repmat(colon(numH*numCatVar + 1, size(xcat, 2) + numContVar), [B, 1]);
                x     = [xcat, xcont];
                idx   = [idx1, idx2];
            else
                x   = xcat;
                idx = idx1;
            end
            xG    = [0*randn(B, latentDim - numH * numCatVar - numContVar),...
                x];
            xG    = reshape(xG', [numel(xG), 1]);
            x     = reshape(x', [numel(x), 1]);
            idx   = dp.selectIndices(idx, B, numH * numCatVar + numContVar,...
                dtype);
            if gpu == 1
                x   = gpuArray(x);
                idx = gpuArray(idx);
                xG  = gpuArray(xG);
            end
        end
        function [x, idx]     = generateLabelsCelebA_plot(numClasses, ...
                numCat, B, dtype)
             x = randi(numClasses, [B, numCat]) - 1; 
%              x= [7     1     5     6     1     2     0     7     2     4
%                  0     2     9     5     9     2     9     5     5     3
%                  2     1     6     5     0     2     1     4     7     0
%                  4     7     6     4     3     5     0     3     8     2
%                  0     0     1     5     0     6     2     0     1     3];
%              x = repmat(x', [10, 1]);
%              x = reshape(x, [10, 10*5])';

            %Gender
%             x(:,8) = repmat([0:9]', [5, 1]);
            % ?
%             x(:,9) = repmat([0:9]', [5, 1]);
            % ?
%             x(:,10) = repmat([0:9]', [5, 1]);
%             % Constrat
%              x(:,7) = repmat([0:9]', [5, 1]);
            % Azimuth
%             x(:,6) = repmat([0:9]', [5, 1]);
            % ?
%             x(:,5) = repmat([0:9]', [5, 1]);
%             %?
%             x(:,4) = repmat([0:9]', [5, 1]);
             % Hair color
%              x(:,3) = repmat([0:9]', [5, 1]);
%              % Background
%              x(:,2) = repmat([0:9]', [5, 1]);
             % Hair style
%              x(:,1) = repmat([0:9]', [5, 1]);

%             x = repmat([0:9], [B, 1]);
%             x = repmat([5     3     0     1     5     9     1     5     0     2], [B, 1]);            
%             x = x*0+2;
            x = reshape(x', [numel(x), 1]);
            x = reshape(x, [numel(x), 1]);
%             x = 0*ones(B, 1);
            [x, idx] = dp.encoder(x, numClasses, dtype);
        end
        function [x, idx]     = generateLabels_plot(numClasses, numCat, B,...
                dtype)
             x = randi(numClasses, [B, numCat]) - 1; 
%              x = [5     3     0     1     5     9     1     5     0     2
%                   5     3     8     6     9     5     1     0     3     7
%                   0     1     8     3     3     0     2     0     3     6
%                   6     6     9     4     4     0     6     1     9     5
%                   6     3     4     1     9     8     4     0     4     6];
%              x = repmat(x', [10, 1]);
%              x = reshape(x, [10, 10*5])';
%             x = repmat([5     7     1     7     2     1     4     3     8     0], [B, 1]); 
%             x = repmat([5     3     0     1     5     9     1     5     0     2], [B, 1]); 
%             %Hair color
%             x(:,8) = repmat([0:9]', [5, 1]);
%             % Hair color
%             x(:,9) = repmat([0:9]', [5, 1]);
%             % Gender
%             x(:,6) = repmat([0:9]', [5, 1]);
%             % Skin color
%              x(:,5) = repmat([0:9]', [5, 1]);
%             % Long hair
%             x(:,3) = repmat([0:9]', [5, 1]);
%             % Smile + rotation
%             x(:,1) = repmat([0:9]', [5, 1]);
%             % Shirt
%             x(:,4) = [0:9]';

%             x = repmat([0:9], [B, 1]);
%             x = repmat([5     3     0     1     5     9     1     5     0     2], [B, 1]);            
%             x = x*0+2;
            x = reshape(x', [numel(x), 1]);
            x = repmat([0:9], [B/10, numCat]);
            x = reshape(x, [numel(x), 1]);
%             x = 0*ones(B, 1);
            [x, idx] = dp.encoder(x, numClasses, dtype);
        end               
               
        % Initialization
        function [net, states, maxIdx, netInfo] = initialization(net)
            % Build indices
            net = indices.initialization(net);
            net = indices.layerEncoder(net);
            net = indices.parameters(net);
            if net.cuda 
                net = indices.covarianceCUDA(net);
            else
                net = indices.covariance(net);
            end
            netInfo = indices.savedInfo(net);
            % States
            states = tagi.initializeStates(net.nodes, net.batchSize, ...
                net.repBatchSize, net.xsc, net.dtype, net.gpu);
            maxIdx = tagi.initializeMaxPoolingIndices(net.nodes, net.layer,...
                net.layerEncoder, net.batchSize, net.repBatchSize, ...
                net.dtype, net.gpu);                        
        end
        function [theta, states, normStat, maxIdx] = extractNet(net)
            theta    = net.theta;
            states   = net.states;
            normStat = net.normStat;
            maxIdx   = net.maxIdx;
        end
        function net = compressNet(net, theta, states, normStat, maxIdx)
            net.theta    = theta;
            net.states   = states;
            net.normStat = normStat;
            net.maxIdx   = maxIdx;
        end
    end
end