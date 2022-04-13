 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         task
% Description:  Run different tasks such as classification, regression, etc
% for the neural networks defined in the config file.
% Authors:      Luong-Ha Nguyen & James-A. Goulet
% Created:      July 02, 2020
% Updated:      August 26, 2021
% Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
% Copyright (c) 2021 Luong-Ha Nguyen & James-A. Goulet. All rights reserved. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
classdef task
    methods (Static)           
        % Reinforcement learning
        function tagiOptNstepQ(netQ, netA, env, trainedModelDir, startEpisode)                        
            % Batch-one nets
            netQ1              = netQ;
            netQ1.batchSize    = 1;
            netQ1.repBatchSize = 1; 
            netA1              = netA;
            netA1.batchSize    = 1;
            netA1.repBatchSize = 1;
            
            % Initialize nets
            [netQ, statesQ, maxIdxQ, netQinfo] = network.initialization(netQ); 
            [netQ1, statesQ1, maxIdxQ1]        = network.initialization(netQ1); 
            [netA, statesA, maxIdxA, netAinfo] = network.initialization(netA); 
            [netA1, statesA1, maxIdxA1]        = network.initialization(netA1); 
            normStatQ1 = tagi.createInitNormStat(netQ1);
            normStatA1 = tagi.createInitNormStat(netA1);
            % Hyperparameters
            modelName    = netQ.modelName;
            dataName     = netQ.dataName;
            cdresults    = netQ.cd;
            savedUpdate  = netQ.savedUpdate;
            nsteps       = netQ.rl.stepUpdate;
            dtype        = netQ.dtype;
            B            = netQ.batchSize;
            rB           = netQ.repBatchSize;
            numEpisodes  = netQ.rl.numEpisodes; 
            maxNumSteps  = netQ.rl.maxNumSteps;
            logInterval  = netQ.logInterval;
            numEp4avg    = 100;
            numDataPerBatch = B*rB;
            
            stat       = zeros(numEpisodes + startEpisode, 5, dtype);
            statS      = zeros(round(maxNumSteps / nsteps), 4, dtype);
            myhist     = zeros(round(maxNumSteps / nsteps), nsteps);
            Syhist     = zeros(round(maxNumSteps / nsteps), nsteps);
            mQhist     = zeros(round(maxNumSteps / nsteps), nsteps);
            baseline   = zeros(round(maxNumSteps / nsteps), 1);
            myMeanhist = zeros(round(maxNumSteps / nsteps), 1);
            oppSign    = zeros(round(maxNumSteps / nsteps), 1);
            nMqhist    = zeros(round(maxNumSteps / nsteps), nsteps);
            nSqhist    = zeros(round(maxNumSteps / nsteps), nsteps);
            dcRewhist  = zeros(round(maxNumSteps / nsteps), nsteps);
            
            % Load pretrained model if available
            if isempty(trainedModelDir) 
                thetaQ    = tagi.initializeWeightBias(netQ); 
                normStatQ = tagi.createInitNormStat(netQ);
                thetaA    = tagi.initializeWeightBias(netA); 
                normStatA = tagi.createInitNormStat(netA);  
                self.mrew = 0;
                self.Srew = 1;
                self.countRew   = 1E-4;
                self.ms         = 0;
                self.Ss         = 1;
                self.count      = 1E-4;
                self.mrewdc     = 0;
                self.Srewdc     = 1;
                self.countRewdc = 1E-4;
            else
                l         = load(trainedModelDir);
                thetaQ    = l.thetaQ;
                normStatQ = l.normStatQ;
                thetaA    = l.thetaA;
                normStatA = l.normStatA;
                stat(1:startEpisode, :) = l.stat(1:startEpisode,:);
                self.mrew  = l.netAinfo.mrew;
                self.Srew  = l.netAinfo.Srew;
                self.countRew   = l.netAinfo.countRew;
                self.ms         = l.netAinfo.ms;
                self.Ss         = l.netAinfo.Ss;
                self.count      = l.netAinfo.count;
                self.mrewdc     = l.netAinfo.mrewdc;
                self.Srewdc     = l.netAinfo.Srewdc;
                self.countRewdc = l.netAinfo.countRewdc;
                clear l
            end 
            
            netQ.trainMode = true;
            netA.trainMode = true;
            
            % Compress nets
            netQ = network.compressNet(netQ, thetaQ, statesQ, normStatQ,...
                maxIdxQ);
            netA = network.compressNet(netA, thetaA, statesA, normStatA,...
                maxIdxA);
            
            netQ1 = network.compressNet(netQ1, thetaQ, statesQ1, normStatQ1,...
                maxIdxQ1);
            netA1 = network.compressNet(netA1, thetaA, statesA1, normStatA1,...
                maxIdxA1);
            
            netQ2 = netQ;
                       
            % Initialize loop 
            loop          = 0; 
            epsCount      = startEpisode;
            epsCountDisp  = epsCount;
            savedloop     = 0;
            run           = true; 
            self.state    = [];
            self.done     = true;
            self.epsRew   = zeros(1, 1, dtype);
            self.epsSteps = zeros(1, 1, dtype);
            self.action   = nan(netA.rl.numActions, 1);
            self.ret      = 0;           
            tt            = 0;
            tts           = 0;   
            
            disp('Training...')
            disp(' ')
            while run
                ts = tic;  
                 % Decay observation nosie 
                if loop>1
                    netQ.sv  = netQ.sv * netQ.svDecayFactor;
                    netA.sv  = netA.sv * netA.svDecayFactor;
                    if netQ.sv < netQ.svmin; netQ.sv = netQ.svmin;end
                    if netA.sv < netA.svmin; netA.sv = netA.svmin;end
                    netQ1.sv = netQ.sv;
                    netA1.sv = netA.sv;
                end
                
%                 % Collect data
                [my, Sy, ~, statelist, actionlist, donelist, epsRewardlist,...
                 epsSteplist, self, nMq, nSq, dcRew] = task.getDatafromEnv(env,...
                 netQ1, netA1, self);                       
                loop       = loop + 1;
                savedloop  = savedloop + 1;
                totalSteps = loop * nsteps;  
                
                % History
                smy                 = gather(mean(my));
                myhist(loop, :)     = gather(my');
                Syhist(loop, :)     = gather(Sy');
                myMeanhist(loop, :) = gather(mean(my));
                nMqhist(loop, :)    = gather(nMq');
                nSqhist(loop, :)    = gather(nSq');
                dcRewhist(loop, :)  = gather(dcRew');
                                                                                       
                % Update policy network
                loopEpoch = 0;
                runOptEpoch = true;              
                while runOptEpoch 
                    thetaQnew = netQ.theta;
                    thetaAnew = netA.theta;
                    loopEpoch = loopEpoch + 1;
                    idxTrain  = randperm(nsteps, nsteps);
                    kloop = 0;
                    numOppSign = 0;
                    for k = 1 : numDataPerBatch :nsteps
                        kloop       = kloop  + 1; 
                        idxBatch    = k : k + numDataPerBatch - 1;
                        idxloop     = idxTrain(idxBatch);
                        stateBatch  = cat(1, statelist{idxloop, 1});
                        actionBatch = cat(1, actionlist{idxloop, 1});
                        myBatch     = my(idxloop);
                        SyBatch     = Sy(idxloop);
                        
                        % Update netQ
                        [dthetaQ, ~, malQ] = network.tagiOptNstepQ(netQ,...
                            netA, stateBatch, actionBatch, myBatch, SyBatch);
                        mQhist(loop, idxloop) = gather(malQ)';
                        thetaQnew = tagi.globalParameterUpdate(thetaQnew,...
                            dthetaQ, netQ.gpu);
                        
                        % Update netA
                        netQ2.theta = tagi.globalParameterUpdate(netQ.theta,...
                            dthetaQ, netQ.gpu);
                        [dthetaA, numOppSignNew] = network.tagiOptA(netQ2,...
                            netA, stateBatch);
                        numOppSign = numOppSign + numOppSignNew;
                        thetaAnew  = tagi.globalParameterUpdate(thetaAnew,...
                            dthetaA, netA.gpu);
                        
                        if mod(kloop, round(netQ.rl.updateBatch / numDataPerBatch)) == 0
                            netQ.theta  = thetaQnew;
                            netA.theta  = thetaAnew;
                            thetaQnew   = netQ.theta;
                            thetaAnew   = netA.theta;
                            netQ2.theta = thetaQnew;
                        end
                    end    
                    netQ.theta  = thetaQnew;
                    netA.theta  = thetaAnew;
                    netQ1.theta = thetaQnew;
                    netA1.theta = thetaAnew;
                    netQ2.theta = thetaQnew;
                    oppSign(loop) = gather(double(numOppSign) ...
                        / double(nsteps * netQ.nodes(end - 1)));
                    if loopEpoch >= netQ.rl.noptepochs; break; end                    
                end                
                te  = toc(ts);
                tts = tts + te;
                tt  = tt + te;
                
                % Save stat
                if any(donelist)
                    sumDones = sum(donelist, 1);
                    stat(epsCount + 1 : epsCount+sumDones, 1) = epsRewardlist(donelist);
                    stat(epsCount + 1 : epsCount+sumDones, 2) = epsSteplist(donelist);
                    stat(epsCount + 1 : epsCount+sumDones, 3) = tts / 60 / ...
                        (savedloop * nsteps) * epsSteplist(donelist); %[min]
                    for d = 1:sumDones
                        epsCount = epsCount+1;
                        stat(epsCount, 4) = mean(stat(max(1, epsCount - numEp4avg) : epsCount, 1));  
                        stat(epsCount, 5) = mean(stat(max(1, epsCount - numEp4avg) : epsCount, 2));  
                    end                
                    
                    % Reset the count
                    tts       = 0;
                    savedloop = 0;
                end
                
                % Save data w.r.t time stpes
                statS(loop, 1) = totalSteps;
                statS(loop, 2) = tt / 60; %[mins]
                statS(loop, 3) = mean(stat(max(1, epsCount - numEp4avg) : epsCount, 1));
                statS(loop, 4) = mean(stat(max(1, epsCount - numEp4avg) : epsCount, 2));
                
                % Display results
                if mod(loop, logInterval) == 0 && epsCount > epsCountDisp
                    tr = tt / (totalSteps) * (maxNumSteps-totalSteps) / 60;
                    formatSpec = 'Step# %4.2e/%0.0e, Reward: %7.1f, Avg. reward: %7.0f, steps: %4.0f, rew/step: %3.1f, Time left: %1.0f mins\n';
                    fprintf(formatSpec, totalSteps, maxNumSteps, ...
                        stat(epsCount, 1), stat(epsCount, 4), ...
                        stat(epsCount, 5), stat(epsCount, 4) / stat(epsCount, 5),...
                        tr);
                    epsCountDisp = epsCount;
                end
                
                % Save models
                if totalSteps >= maxNumSteps
                    stat = stat(1:epsCount, :);
                end
                if mod(loop, savedUpdate)==0 || totalSteps >= maxNumSteps
                    netAinfo.numParamsPerlayer_2 = netA.numParamsPerlayer_2;
                    netAinfo.ms         = self.ms;
                    netAinfo.Ss         = self.Ss;
                    netAinfo.count      = self.count;
                    netAinfo.mrew       = self.mrew;
                    netAinfo.Srew       = self.Srew;
                    netAinfo.countRew   = self.countRew;
                    netAinfo.mrewdc     = self.mrewdc;
                    netAinfo.Srewdc     = self.Srewdc;
                    netAinfo.countRewdc = self.countRewdc;                   
                    netAinfo.oppSign    = oppSign;
                    
                    netQinfo.myhist     = myhist;
                    netQinfo.Syhist     = Syhist;
                    netQinfo.mQhist     = mQhist;
                    netQinfo.baseline   = baseline;
                    netQinfo.myMeanhist = myMeanhist;
                    netQinfo.nMqhist    = nMqhist;
                    netQinfo.nSqhist    = nSqhist;
                    netQinfo.dcRewhist  = dcRewhist;                   
                    task.saveRLDQN2net(cdresults, modelName, dataName,...
                        netQ.theta, netQ.normStat, netQinfo, netA.theta,...
                        netA.normStat, netAinfo, statS, ...
                        epsCount + startEpisode)
                end
                
                % Stop               
                if totalSteps >= maxNumSteps; break; end
            end          
            disp('Done.')
        end 

            % Hyperparameters
            modelName    = netP.modelName;
            dataName     = netP.dataName;
            cdresults    = netP.cd;
            savedUpdate  = netP.savedUpdate;
            nsteps       = netP.rl.stepUpdate;
            dtype        = netP.dtype;
            B            = netP.batchSize;
            rB           = netP.repBatchSize;
            numEpisodes  = netP.rl.numEpisodes; 
            maxNumSteps  = netP.rl.maxNumSteps;
            logInterval  = netP.logInterval;
            gpu          = netP.gpu;
            fire         = py.list({'FIRE'});
            numEp4avg    = 100;
            numDataPerBatch = B*rB;
            
            % Batch-one nets
            netP1              = netP;
            netP1.batchSize    = 1;
            netP1.repBatchSize = 1; 
            
            % Initialize nets
            [netP, statesP, maxIdxP, netPinfo] = network.initialization(netP); 
            [netP1, statesP1, maxIdxP1] = network.initialization(netP1); 
            normStatP1 = tagi.createInitNormStat(netP1);
            stat       = zeros(numEpisodes + startEpisode, 6, dtype);
            
            myhist     = zeros(round(maxNumSteps / nsteps), nsteps);
            Syhist     = zeros(round(maxNumSteps / nsteps), nsteps);
            mQhist     = zeros(round(maxNumSteps / nsteps), nsteps);
            myMeanhist = zeros(round(maxNumSteps / nsteps), 1);
            nMqhist    = zeros(round(maxNumSteps / nsteps), nsteps);
            nSqhist    = zeros(round(maxNumSteps / nsteps), nsteps);
            dcRewhist  = zeros(round(maxNumSteps / nsteps), nsteps);
            
            % Load pretrained model if available
            if isempty(trainedModelDir) 
                thetaP    = tagi.initializeWeightBias(netP); 
                normStatP = tagi.createInitNormStat(netP); 
                self.ms         = 0;
                self.Ss         = 1;
                self.countS     = 1E-4;
                self.mrew       = 0;
                self.Srew       = 1;
                self.countRew   = 1E-4;
                self.mrewdc     = 0;
                self.Srewdc     = 1;
                self.countRewdc = 1E-4;
            else
                l         = load(trainedModelDir);
                thetaP    = l.theta;
                normStatP = l.normStat;
                stat(1 : startEpisode, :) = l.stat(end - numEp4avg + 1 : end, :);
                clear l
            end            
            netP.trainMode = true;
            
            % Atari
            meanings = env.get_action_meanings();
            for m = 1:length(meanings)
                isfire = meanings(m)==fire;
                if isfire; break;end
            end
            self.isfire = isfire&&length(meanings)>3;
            
            % Initialize loop  
            loop            = 0;   
            epsCount        = startEpisode;
            epsCountDisp    = epsCount;
            savedloop       = 0;
            run             = true;   
            frameCount      = 0;
            epTotalSteps    = 0;
            self.savedFrame = cell(5000, 1);
            self.done       = true;
            self.stateM     = zeros(netP.rl.imgSize, dtype);
            self.epsRew     = 0;
            self.epsSteps   = 0;
            self.action     = nan(netP.rl.numActions, 1);
            self.savedFrame = cell(nsteps * netP.rl.numFrameskip, 1);
            
            disp('Training...')
            disp(' ')
            tt  = 0;
            tts = 0;
            while run
                ts = tic;
                % Decay observertion noise sv
                if mod(loop, 1)==0
                    netP.sv = netP.sv * netP.svDecayFactor;
                    if netP.sv < netP.svmin; netP.sv = netP.svmin;end
                    netP1.sv = netP.sv;
                end
                [my, Sy, statelist, actionlist, donelist, epsRewlist,...
                 epsSteplist, self, dcRew, nMq, nSq] = task.getDatafromEnvAtariCUDA(env,...
                 netP1, thetaP, statesP1, normStatP1, maxIdxP1, self);        
                loop       = loop + 1;
                savedloop  = savedloop + 1;
                totalSteps = loop * nsteps;
                
                % History
                myhist(loop, :)     = gather(my');
                Syhist(loop, :)     = gather(Sy');
                myMeanhist(loop, :) = gather(mean(my));
                nMqhist(loop, :)    = gather(nMq');
                nSqhist(loop, :)    = gather(nSq');
                dcRewhist(loop, :)  = gather(dcRew');
                
                
                % Save frames
                if any(donelist)
                    frameCount = 0;
                end
                numFrameLoop = size(self.savedFrame, 1);
                frameCount = frameCount + numFrameLoop;
                self.savedFrame(frameCount - numFrameLoop + 1 : frameCount, 1) = self.savedFrame;
                                                                       
                % Update policy network
                runOptEpoch = true;
                loopEpoch = 0;
                while runOptEpoch
                    loopEpoch = loopEpoch + 1;
                    idxTrain = 1 : nsteps;
                    for k = 1 : numDataPerBatch : nsteps
                        idxBatch    = k : k + numDataPerBatch - 1;
                        idxloop     = idxTrain(idxBatch);
                        stateBatch  = cat(1, statelist{idxloop, 1});
                        actionBatch = actionlist(idxloop);
                        myBatch     = my(idxloop);
                        SyBatch     = Sy(idxloop);
                        [dthetaP, normStatP,...
                            malP] = network.nstepQ1netCUDA(netP, thetaP,...
                            normStatP, statesP, maxIdxP, stateBatch,...
                            actionBatch, myBatch, SyBatch);
                        mQhist(loop, idxBatch) = gather(malP)';
                        thetaP = tagi.globalParameterUpdate(thetaP, dthetaP,...
                        gpu);
                    end
                    if loopEpoch >= netP.rl.noptepochs; break; end   
                end
                te  = toc(ts);
                tts = tts + te;
                tt  = tt + te;
                
                % Save stat
                if any(donelist)
                    sumDones = sum(donelist, 1);
                    stat(epsCount+1:epsCount+sumDones, 1) = epsRewlist(donelist);
                    stat(epsCount+1:epsCount+sumDones, 2) = epsSteplist(donelist);
                    stat(epsCount+1:epsCount+sumDones, 3) = tts / 60 ...
                        / (savedloop * nsteps) * epsSteplist(donelist);
                    epTotalSteps = epTotalSteps + ...
                        sum(stat(epsCount + 1 : epsCount + sumDones, 2), 1);
                    for d = 1 : sumDones
                        epsCount = epsCount+1;
                        stat(epsCount, 4) = mean(stat(max(1, epsCount - numEp4avg) : epsCount, 1));  
                        stat(epsCount, 5) = mean(stat(max(1, epsCount - numEp4avg) : epsCount, 2));  
                    end                
                    
                     % Reset the count
                    self.savedFrame = cell(5000, 1);
                    tts         = 0;
                    savedloop   = 0;
                    frameCount  = 0;                    
                end
                
                % Display results
                if mod(loop, logInterval) == 0 ...
                        && epsCount > epsCountDisp
                    tr = tt / (totalSteps) * (maxNumSteps-totalSteps) / 60;
                    formatSpec = 'Step# %4.2e/%0.0e, Reward: %7.1f, Avg. reward: %7.1f, steps: %7.0f, Time left: %1.0f mins\n';
                    fprintf(formatSpec, totalSteps, maxNumSteps, ...
                        stat(epsCount, 1), stat(epsCount, 4), stat(epsCount, 5), tr);
                    epsCountDisp = epsCount;
                end                              
                
                % Save models
                if epTotalSteps >= maxNumSteps
                    stat = stat(1 : epsCount, :);
                end
                if mod(loop, savedUpdate) == 0 ...
                        || epTotalSteps >= maxNumSteps
                    task.saveRL1net(cdresults, modelName, dataName, thetaP,...
                        normStatP,  stat, netPinfo, epsCount);
                end
                
                % Stop               
                if epTotalSteps>=maxNumSteps; break; end
            end          
            disp('Done.')
        end 
                
        function avgReward = test1net(net, env, trainedModelDir)
             % Initialization
            np = py.importlib.import_module('numpy');
            numEpisodes = net.rl.numEpisodes;
            [net, netStates, maxIdx] = network.initialization(net); 
            
             % Load trained model
             if isempty(trainedModelDir)
                 theta    = tagi.initializeWeightBias(net);
                 normStat = tagi.createInitNormStat(net);
             else
                 l         = load(trainedModelDir);
                 theta    = l.theta;
                 normStat = l.normStat;
                 clear l
             end           
            savedFrame = cell(5000, 1);
            stat       = zeros(numEpisodes, 2, net.dtype);
            for ep = 1 : numEpisodes
                obs   = env.reset(); 
                state = obs(:);
                
                % Initialize loop
                run  = 1;
                ts   = tic;
                te   = 0;
                loop = 0;
                episodeReward = 0;                
                while run
                    loop = loop + 1;
                    
                    % Action
                    action = rltagi.getAction1net(net, theta, netStates,...
                        normStat, maxIdx, state);
                    action = gather(action);                   
                    
                    % Collect observation
                    action2gym = action - 1;
                    screen = env.render();
                    screen = single(np.array(screen));
                    savedFrame{loop} = screen;
                    [obs, reward, done] = env.step(action2gym);
                    
                    % Reward
                    episodeReward = episodeReward + reward;
                    if ~done
                        nextState = obs(:);
                    else
                        nextState = zeros(size(state));
                    end 
                    state = nextState;
                    if done; break; end
                end                
                te = te + toc(ts);
                tr = te/ep*(numEpisodes-ep)/60;
                
                stat(ep, 1) = episodeReward;
                stat(ep, 2) = mean(stat(max(1,ep-100):ep, 1));
                formatSpec  = 'Episode# %3.0f/%0.0f, Reward: %7.1f, Avg. reward: %7.1f, Time left: %1.0f mins\n';
                fprintf(formatSpec, ep, numEpisodes, stat(ep, 1),...
                    stat(ep, 2), tr)
                rltagi.video(savedFrame(1:loop),...
                    [net.dataName '_episode_' num2str(ep) '_R' num2str(episodeReward)])
            end
            env.close();
            avgReward = stat(end, 2);
            disp('Done.')
        end
        function avgReward = testContAction1net(net, env, trainedModelDir)
             % Initialization
             actionLow   = net.rl.actionLow;
             actionHigh  = net.rl.actionHigh;
             np = py.importlib.import_module('numpy');
             numEpisodes = net.rl.numEpisodes;
             [net, netStates, maxIdx] = network.initialization(net);
            
             % Load trained model
             if isempty(trainedModelDir)
                 theta    = tagi.initializeWeightBias(net);
                 normStat = tagi.createInitNormStat(net);
             else
                 l        = load(trainedModelDir);
                 theta    = l.thetaA;
                 normStat = l.normStatA;
                 ms       = l.netAinfo.ms;
                 Ss       = l.netAinfo.Ss;
                 clear l
             end 
            net = network.compressNet(net, theta, netStates, normStat, maxIdx);
            stat       = zeros(numEpisodes, 4, net.dtype);
            for ep = 1 : numEpisodes
                obs   = env.reset(); 
                state = obs(:);
                state = (state - ms) ./ sqrt(Ss + 1E-8);
                state = rltagi.clamp(state, net.rl.obsLow, net.rl.obsHigh);
                
                % Initialize loop
                run  = 1;
                ts   = tic;
                te   = 0;
                loop = 0;
                episodeReward = 0;                
                while run
                    loop = loop + 1;
                    
                    % Action
                    [action] = rltagi.getContAction1net(net, state);
                    action2gym = gather(abs(actionHigh) .* tanh(action));
                    action = gather(action2gym);                   
                    
                    % Collect observation
%                     screen = env.render();
%                     screen = single(np.array(screen));
%                     savedFrame{loop} = screen;
                    [obs, reward, done] = env.step(action);
                    
                    % Reward
                    episodeReward = episodeReward + reward;
                    if ~done
                        nextState = obs(:);
                        nextState = (nextState - ms) ./ sqrt(Ss + 1E-8);
                        nextState = rltagi.clamp(nextState, net.rl.obsLow,...
                            net.rl.obsHigh);
                    else
                        nextState = zeros(size(state));
                    end 
                    state = nextState;
                    if done; break; end
                end                
                te = te + toc(ts);
                tr = te/ep*(numEpisodes-ep)/60;
                
                stat(ep, 1) = episodeReward;
                stat(ep, 2) = loop;
                stat(ep, 3) = mean(stat(max(1,ep-100):ep, 1));
                stat(ep, 4) = mean(stat(max(1,ep-100):ep, 2));
                formatSpec  = 'Episode# %3.0f/%0.0f, Reward: %7.1f, Avg. reward: %7.1f, Avg. len: %7.1f, Time left: %1.0f mins\n';
                fprintf(formatSpec, ep, numEpisodes, stat(ep, 1),...
                    stat(ep, 3), stat(ep, 4), tr)
%                 rltagi.video(savedFrame(1:loop), [net.dataName '_episode_' num2str(ep) '_R' num2str(episodeReward)])
            end
            env.close();
            avgReward = stat(end, 2);
            disp('Done.')
        end
        function avgReward = testContAction1net_V2(net, env, trainedModelDir)
             % Initialization
             actionLow   = net.rl.actionLow;
             actionHigh  = net.rl.actionHigh;
             np = py.importlib.import_module('numpy');
             numEpisodes = net.rl.numEpisodes;
             [net, netStates, maxIdx] = network.initialization(net);
            
             % Load trained model
             if isempty(trainedModelDir)
                 theta    = tagi.initializeWeightBias(net);
                 normStat = tagi.createInitNormStat(net);
             else
                 l        = load(trainedModelDir);
                 theta    = l.thetaA;
                 normStat = l.normStatA;
                 ms       = 0;
                 Ss       = 1;
                 clear l
             end 
            
            net = network.compressNet(net, theta, netStates, normStat, maxIdx);
            savedFrame = cell(5000, 1);
            stat       = zeros(numEpisodes, 4, net.dtype);
            for ep = 1:numEpisodes
                obs   = env.reset(); 
                state = obs(:);
                state = (state - ms) ./ sqrt(Ss + 1E-8);
                state = rltagi.clamp(state, net.rl.obsLow, net.rl.obsHigh);
                
                % Initialize loop
                run  = 1;
                ts   = tic;
                te   = 0;
                loop = 0;
                episodeReward = 0;                
                while run
                    loop = loop + 1;
                    
                    % Action
                    [action] = rltagi.getContAction1net(net, state);
                    action2gym = gather(abs(actionHigh).*tanh(action));
                    action = gather(action2gym);                   
                    
                    % Collect observation
%                     screen = env.render();
%                     screen = single(np.array(screen));
%                     savedFrame{loop} = screen;
                    [obs, reward, done] = env.step(action);
                    
                    % Reward
                    episodeReward = episodeReward + reward;
                    if ~done
                        nextState = obs(:);
                        nextState = (nextState - ms) ./ sqrt(Ss + 1E-8);
%                         nextState = rltagi.clamp(nextState, net.rl.obsLow, net.rl.obsHigh);
                    else
                        nextState = zeros(size(state));
                    end 
                    state = nextState;
                    if done; break; end
                end                
                te = te + toc(ts);
                tr = te/ep*(numEpisodes-ep)/60;
                
                stat(ep, 1) = episodeReward;
                stat(ep, 2) = loop;
                stat(ep, 3) = mean(stat(max(1,ep-100):ep, 1));
                stat(ep, 4) = mean(stat(max(1,ep-100):ep, 2));
                formatSpec  = 'Episode# %3.0f/%0.0f, Reward: %7.1f, Avg. reward: %7.1f, Avg. len: %7.1f, Time left: %1.0f mins\n';
                fprintf(formatSpec, ep, numEpisodes, stat(ep, 1),...
                    stat(ep, 3), stat(ep, 4), tr)
%                 rltagi.video(savedFrame(1:loop), [net.dataName '_episode_' num2str(ep) '_R' num2str(episodeReward)])
            end
            env.close();
            avgReward = stat(end, 2);
            disp('Done.')
        end       
        function [my, Sy, mbdone, statelist, actionlist, donelist, epsRewlist,...
                epsSteplist, self, nMq, nSq, dcRew] = getDatafromEnv(env,...
                netQ, netA, self)
            % Initialization
            gpu         = netQ.gpu;          
            gamma       = netQ.rl.gamma;
            dtype       = netQ.dtype;
            nsteps      = netQ.rl.stepUpdate;
            numStepsEps = netQ.rl.numStepsEps;
            actionHigh  = netQ.rl.actionHigh;
            
            statelist   = cell(nsteps, 1); 
            actionlist  = cell(nsteps, 1);
            donelist    = zeros(nsteps, 1, 'logical');
            epsRewlist  = zeros(nsteps, 1, dtype);
            epsSteplist = zeros(nsteps, 1, dtype);
            mbdone      = zeros(nsteps + 1, 1, 'logical');
            if gpu
                rewardlist = zeros(nsteps, 1, dtype, 'gpuArray');
                my    = zeros(nsteps, 1, dtype, 'gpuArray');
                Sy    = zeros(nsteps, 1, dtype, 'gpuArray');
                dcRew = zeros(nsteps, 1, dtype, 'gpuArray');
            else
                rewardlist = zeros(nsteps, 1, dtype);
                my    = zeros(nsteps, 1, dtype);
                Sy    = zeros(nsteps, 1, dtype);
                dcRew = zeros(nsteps, 1, dtype);
            end
            
            % Reset environment
            if self.done
                obs    = env.reset(); 
                state  = obs(:); 
                [self.ms, self.Ss, ...
                    self.count] = rltagi.runningMeanStd_V2(state, self.ms,...
                    self.Ss, self.count, 1);
                state = (state - self.ms) ./ (sqrt(self.Ss) + 1E-8);
                state = rltagi.clamp(state, netQ.rl.obsLow, netQ.rl.obsHigh);
                self.ret      = 0;
                self.epsSteps = 0;
                self.epsRew   = 0; 
            else
                state = self.state;                
            end 
            mbdone(1) = self.done;
            if gpu; state = gpuArray(state); end   
            
            % Initialize loop
            loop    = 0;
            episode = 1;
            loopUd  = 0;
            for s = 1 : nsteps
                loop   = loop + 1;
                loopUd = loopUd + 1;
                self.epsSteps = self.epsSteps + 1;
                statelist{s}  = state;                                
                
                % Select action
                if any(isnan(self.action))
                    [action] = rltagi.getContAction1net(netA, state);
                else
                    action = self.action;
                    self.action = nan(netA.rl.numActions, 1); 
                end
                actionlist{s} = action;                                                             
                
                % Take action in environment
                action2gym = gather(abs(actionHigh) .* tanh(action));
                [obs, reward, done] = env.step(action2gym);
                mbdone(s + 1) = done;
                
                % Normalize reward
                self.epsRew = self.epsRew + reward; 
                self.ret    = gamma * self.ret + reward;
                [self.mrew, self.Srew, ...
                    self.countRew] = rltagi.runningMeanStd_V2(self.ret, ...
                    self.mrew, self.Srew, self.countRew, 1);
                if self.countRew > 1 
                    reward = reward ./ (sqrt(self.Srew) + 1E-8);
                    reward = rltagi.clamp(reward, netQ.rl.rewardLow, ...
                        netQ.rl.rewardHigh);
                end                
                if gpu; rewardlist(s) = gpuArray(reward); else; rewardlist(s) = reward; end    
                
                % Next step
                if ~done
                    nextState = obs(:);
                    [self.ms, self.Ss, ...
                        self.count] = rltagi.runningMeanStd_V2(nextState, ...
                        self.ms, self.Ss, self.count, 1);
                    nextState = (nextState - self.ms) ./ (sqrt(self.Ss) + 1E-8);
                    nextState = rltagi.clamp(nextState, netQ.rl.obsLow, ...
                        netQ.rl.obsHigh);
                else
                    nextState = nan;
                end                 
                state = nextState;
                if gpu; state = gpuArray(state); nextState = gpuArray(nextState); end       
                
                % Collect data
                if mod(loop, nsteps) == 0 || done
                    % Get next Q
                    if ~done
                        [nextAction]     = rltagi.getContAction1net(netA,...
                            nextState);
                        [lastMq, lastSq] = rltagi.nstepValue_V2(netQ, netA,...
                            nextState, nextAction);
                        [my(s-loopUd+1:s), ...
                            Sy(s-loopUd+1:s)] = rltagi.discountValue(lastMq,...
                            lastSq, gamma, loopUd);
                        self.action = nextAction;
                    else
                        self.action = nan(netA.rl.numActions, 1);                        
                    end
                    
                    % Discount rewards
                    dcRewloop = rltagi.discountReward(rewardlist(s - loopUd + 1 : s),...
                        gamma, loopUd);                   
                    dcRew(s - loopUd + 1 : s) = dcRewloop;  
                    loopUd = 0;                   
                end
                
                % Update stats for episodes
                donelist(episode)    = done;
                epsRewlist(episode)  = self.epsRew;
                epsSteplist(episode) = self.epsSteps;
                
                % Reset environment
                if (done || self.epsSteps >= numStepsEps) && s < nsteps   
                    obs      = env.reset(); 
                    state    = obs(:);  
                    [self.ms, self.Ss, ...
                        self.count] = rltagi.runningMeanStd_V2(state,...
                        self.ms, self.Ss, self.count, 1);
                    state = (state - self.ms) ./ (sqrt(self.Ss) + 1E-8);
                    state = rltagi.clamp(state, netQ.rl.obsLow, ...
                        netQ.rl.obsHigh);
                    if gpu; state = gpuArray(state); end    
                    
                    % Reset counters
                    self.ret      = 0;
                    self.epsRew   = 0;
                    self.epsSteps = 0;
                    episode = episode + 1;
                end
            end 
            % Reward normalization
            nMq = my;
            nSq = Sy;
            my  = my + dcRew;
                                    
            % Only take the results after having reset
            self.done   = done;
            self.state  = state;
            donelist    = donelist(1:episode);
            epsRewlist  = epsRewlist(1:episode);
            epsSteplist = epsSteplist(1:episode);
        end 
        function [my, Sy, statelist, actionlist, donelist, epsRewlist,...
                epsSteplist, self, discountRew, nMq, nSq] = getDatafromEnvAtariCUDA(env,...
                netP, thetaP, statesP, normStatP, maxIdxP, self)
            % Initialization
            gpu          = netP.gpu;
            numFrames    = netP.rl.numFrames;
            numFrameskip = netP.rl.numFrameskip;            
            gamma        = netP.rl.gamma;
            dtype        = netP.dtype;
            nsteps       = netP.rl.stepUpdate;
            numStepsEps  = netP.rl.numStepsEps;
            resizeMethod = netP.rl.resizeMethod;
            imgSize      = netP.imgSize;
            
            statelist   = cell(nsteps, 1);
            actionlist  = zeros(nsteps, 1, dtype);
            rewardlist  = zeros(nsteps, 1, dtype);
            donelist    = zeros(nsteps, 1, 'logical');
            epsRewlist  = zeros(nsteps, 1, dtype);
            epsSteplist = zeros(nsteps, 1, dtype);
            self.savedFrame  = cell(nsteps*numFrameskip, 1);
            if gpu
                discountRew = zeros(nsteps, 1, dtype, 'gpuArray');
                my = zeros(nsteps, 1, dtype, 'gpuArray');
                Sy = zeros(nsteps, 1, dtype, 'gpuArray');
            else
                discountRew = zeros(nsteps, 1, dtype);
                my = zeros(nsteps, 1, dtype);
                Sy = zeros(nsteps, 1, dtype);
            end
            % Reset environment
            if self.done
                stateM = zeros(imgSize, dtype);
                if self.isfire
                    [stateM(:,:,end),...
                        ~, lives] = wrappers.noopFireResetEnv_V3(env,...
                        numFrameskip, resizeMethod); 
                else
                    [obs, ~, ~, info] = wrappers.noopResetEnv(env);
                    img    = wrappers.atariPreprocessing(obs, resizeMethod);
                    stateM = repmat(cast(img, dtype)/255, [1 1 numFrames]);
                    lives  = info;
                end                                            
                self.epsRew   = 0; 
                self.epsSteps = 0;
            else
                lives = double(env.episodicLife());
                stateM = self.stateM;
            end
            state  = reshape(stateM, [numel(stateM), 1]);
            if gpu; state = gpuArray(state); end   
            
            % Initialize loop
            loopUd  = 0;
            loop    = 0;
            episode = 1;
            frameCount = 0;
            for s = 1:nsteps
                loop     = loop + 1;
                loopUd   = loopUd + 1;
                self.epsSteps = self.epsSteps + 1;
                statelist{s} = state;
                
                % Select action
                if any(isnan(self.action))
                    action = rltagi.getAction1netCUDA(netP, thetaP, statesP,...
                        normStatP, maxIdxP, state);
                else
                    action = self.action;
                    self.action = nan(netP.rl.numActions, 1);
                end
                action = gather(action);
                actionlist(s) = action;
                
                % Take action in environment
                action2gym = gather(action) - 1;
                [nextStateM, rewardclip, reward, obs,...
                    done, info] = wrappers.stackFrames(env, action2gym,...
                    stateM, lives, numFrames, numFrameskip, resizeMethod);
                
                % Normalize reward
                if netP.rl.rewardScaling
                    self.ret    = netP.rl.gamma * self.ret + rewardclip;
                    [self.mrew, self.Srew,...
                        self.countRew] = rltagi.runningMeanStd_V2(self.ret,...
                        self.mrew, self.Srew, self.countRew, 1);
                    if self.countRew > 1
                        rewardclip = rewardclip ./ (sqrt(self.Srew) + 1E-8);
                        rewardclip = rltagi.clamp(rewardclip, ...
                            netP.rl.rewardLow, netP.rl.rewardHigh);
                    end
                end
                rewardlist(s) = rewardclip;
                
                % Lost life?
                newlives = info;
                if newlives < lives && newlives > 0
                    lostlife = true;
                else
                    lostlife = false;
                end   
                
                % Save frame
                numFrameLoop = size(obs, 1);
                frameCount = frameCount + numFrameLoop;
                self.savedFrame(frameCount + 1 - numFrameLoop : frameCount, 1) = obs;
                
                % Reward
                self.epsRew = self.epsRew + reward;
                if ~done && ~lostlife
                    nextState = reshape(nextStateM, [numel(nextStateM), 1]);
                else                   
                    nextState = nan;
                end
                state  = nextState;
                stateM = nextStateM;
                if gpu; state = gpuArray(state); nextState = gpuArray(nextState); end
                
                % Collect data
                if mod(loop, nsteps)==0 || done || lostlife
                    % Get next Q
                    if ~done && ~lostlife
                        [lastMq, lastSq, ...
                            self.action] = rltagi.nstepValue1net(netP, ...
                            thetaP, statesP, normStatP, maxIdxP, nextState);
                        [my(s - loopUd + 1 : s), ...
                         Sy(s - loopUd + 1 : s)] = rltagi.discountValue(lastMq,...
                         lastSq, gamma, loopUd);
                    else
                        self.action = nan(netP.rl.numActions, 1);
                    end
                    
                    % Discount rewards
                    discountRewloop = rltagi.discountReward(rewardlist(s - loopUd + 1 : s),...
                        gamma, loopUd);
                    discountRew(s - loopUd + 1 : s) = discountRewloop;  
                    loopUd = 0;                   
                end
                
                % Episodic life
                if lostlife && ~done
                    runEplife = true;
                    stateM = zeros(imgSize, dtype);
                    while runEplife
                        [stateM(:,:,end), obs, ~, done,...
                            info] = wrappers.episodicLifeResetEnv(env,...
                            newlives, numFrameskip, resizeMethod);  
                        
                        % Save frame
                        numFrameLoop = size(obs, 1);
                        frameCount   = frameCount + numFrameLoop;
                        self.savedFrame(frameCount - numFrameLoop + 1 : frameCount, 1) = obs;
                        
                        if newlives==info||done
                            lives = info;
                            break;
                        else
                            newlives = info;
                        end
                    end
                    state = reshape(stateM, [numel(stateM), 1]);
                    if gpu; state = gpuArray(state); end
                end
                
                % Reset environment
                donelist(episode)    = done;
                epsRewlist(episode)  = self.epsRew;
                epsSteplist(episode) = self.epsSteps;
                
                if (done || self.epsSteps >= numStepsEps) && s < nsteps  
                    stateM = zeros(imgSize, dtype);
                    if self.isfire
                        [stateM(:,:,end), ~,...
                            lives] = wrappers.noopFireResetEnv_V3(env,...
                            numFrameskip, resizeMethod); 
                    else
                        [obs, ~, ~, info] = wrappers.noopResetEnv(env);
                        img    = wrappers.atariPreprocessing(obs, resizeMethod);
                        stateM = repmat(cast(img, dtype)/255, [1 1 numFrames]);
                        lives  = info;
                    end                                     
                    state  = reshape(stateM, [numel(stateM), 1]);
                    if gpu; state = gpuArray(state); end                   
                    
                    % Reset the count
                    self.savedFrame = cell(nsteps*numFrameskip, 1);
                    self.epsRew     = 0;                    
                    self.epsSteps   = 0;
                    episode         = episode + 1;
                    frameCount      = 0;
                end
            end 
            % Normalize rewards
            self.mrewdc           = mean(discountRew);
            stdRewdc              = std(discountRew);
            stdRewdc(stdRewdc==0) = 1;
            self.Srewdc           = stdRewdc.^2;
            discountRew = (discountRew - self.mrewdc) ...
                ./ (sqrt(self.Srewdc) + 1E-12);
            nMq = my;
            nSq = Sy;
            my  = my + discountRew; 
            
            % Only take the results after having reset
            self.savedFrame = self.savedFrame(1 : frameCount);
            self.done   = done;
            self.stateM = stateM;
            donelist    = donelist(1:episode);
            epsRewlist  = epsRewlist(1:episode);
            epsSteplist = epsSteplist(1:episode);
        end
                                                   
        % Adversarial attack
        function adversarialAttack(net, imdb , trainedModelDir)
            % Initialization          
            cdresults  = net.cd;
            modelName  = net.modelName;
            dataName   = net.dataName;
            savedEpoch = net.savedEpoch;
            maxEpoch   = net.maxEpoch;
            NiterTrain = round(imdb.numImages /...
                (net.batchSize * net.repBatchSize));
            NiterVal   = round(imdb.numImages * ...
                (net.numClasses - 1) / (net.batchSize * net.repBatchSize));
            imgStat    = reshape(net.imgStat', [1, 1, size(net.imgStat, 2) 2]);
            net.imgStat= imgStat;
            imdb.imgStat = imgStat;
            truelabels = repmat(reshape(imdb.digitlabels, ...
                [net.batchSize * net.repBatchSize, NiterTrain]), ...
                [(net.numClasses - 1), 1]);
            truelabels = truelabels(:);
            Sx         = cast(net.Sx, net.dtype);
            if net.gpu
                Sx = gpuArray(Sx);
            end
            
            % Encoder Idx
            addIdx   = reshape(repmat(colon(0, net.ny, ...
                (net.batchSize*net.repBatchSize - 1) * net.ny),...
                [net.numClasses, 1]),...
                [net.numClasses * net.batchSize * net.repBatchSize, 1]);
            classObs = repmat(imdb.classObs, [net.batchSize * net.repBatchSize, 1]);
            classIdx = repmat(imdb.classIdx, ...
                [net.batchSize * net.repBatchSize, 1]) +...
                cast(addIdx, class(imdb.classIdx));
            
            % Initialize network
            [net, states, maxIdx, netInfo] = network.initialization(net); 
            if isempty(trainedModelDir)                  
                theta    = tagi.initializeWeightBias(net); 
                normStat = tagi.createInitNormStat(net); 
            else
                l        = load(trainedModelDir);
                theta    = l.theta;
                normStat = l.normStat;
            end
            % Training
            net.trainMode = false;           
            ts = tic;
            disp(' ')
            disp('Generating adversarial samples... ')
            
            % Generate adversarial samples
            if net.cuda
                [advImg, advlabels] = network.adversarialGenerationCUDA(net,...
                    theta, states, normStat, maxIdx, imdb, Sx, classObs,...
                    classIdx, NiterTrain);
            else
                [advImg, advlabels] = network.adversarialGeneration(net,...
                    theta, states, normStat, maxIdx, imdb, Sx, classObs,...
                    classIdx, NiterTrain);
            end
             
            % Attack network
            disp(' ')
            disp('Attacking network... ')
            if net.cuda
                [Pr, sr, er] = network.adversarialAttackRateCUDA(net, theta,...
                    states, normStat, maxIdx, advImg, advlabels, truelabels,...
                    classObs, classIdx, NiterVal);
            else
                [Pr, sr, er] = network.adversarialAttackRate(net, theta,...
                    states, normStat, maxIdx, advImg, advlabels, truelabels,...
                    classObs, classIdx, NiterVal);
            end
            te = toc(ts) / 60;
            
            % Display results
            if net.displayMode == 1
                formatSpec = 'Success rate: %3.1f%%, Gobal success rate: %3.1f%%, Total time: %1.0f mins\n';
                fprintf(formatSpec, 100 - 100 * mean(sr), 100 * mean(er), te);
            end
            disp('Done.')
        end
        function adversarialAttack_V2(Anet, Dnet, imdb , trainedModelDirA,...
                trainedModelDirD)
            % Initialization          
            cdresults  = Anet.cd;
            modelName  = Anet.modelName;
            dataName   = Anet.dataName;
            savedEpoch = Anet.savedEpoch;
            maxEpoch   = Anet.maxEpoch;
            NiterTrain = round(imdb.numImages /...
                (Anet.batchSize * Anet.repBatchSize));
            NiterVal   = round(imdb.numImages * ...
                (Anet.numClasses - 1) / (Anet.batchSize * Anet.repBatchSize));
            imgStat    = reshape(Anet.imgStat', [1, 1, size(Anet.imgStat, 2) 2]);
            Anet.imgStat = imgStat;
            Dnet.imgStat = imgStat;
            imdb.imgStat = imgStat;
            truelabels = repmat(reshape(imdb.digitlabels, ...
                [Anet.batchSize * Anet.repBatchSize, NiterTrain]), ...
                [(Anet.numClasses - 1), 1]);
            truelabels = truelabels(:);
            Sx         = cast(Anet.Sx, Anet.dtype);
            if Anet.gpu
                Sx = gpuArray(Sx);
            end
            
            % Encoder Idx
            addIdx   = reshape(repmat(colon(0, Anet.ny,...
                (Anet.batchSize * Anet.repBatchSize - 1) * Anet.ny),...
                [Anet.numClasses, 1]),...
                [Anet.numClasses * Anet.batchSize * Anet.repBatchSize, 1]);
            classObs = repmat(imdb.classObs, [Anet.batchSize * Anet.repBatchSize, 1]);
            classIdx = repmat(imdb.classIdx, ...
                [Anet.batchSize * Anet.repBatchSize, 1]) + ...
                cast(addIdx, class(imdb.classIdx));
            
            % Initialize network
            [Anet, statesA, maxIdxA, AnetInfo] = network.initialization(Anet); 
            [Dnet, statesD, maxIdxD, DnetInfo] = network.initialization(Dnet); 
            if isempty(trainedModelDirA)                  
                thetaA    = tagi.initializeWeightBias(Anet); 
                normStatA = tagi.createInitNormStat(Anet); 
                thetaD    = tagi.initializeWeightBias(Dnet); 
                normStatD = tagi.createInitNormStat(Dnet); 
            else
                lA        = load(trainedModelDirA);
                thetaA    = lA.theta;
                normStatA = lA.normStat;
                lD        = load(trainedModelDirD);
                thetaD    = lD.theta;
                normStatD = lD.normStat;
            end
            % Training
            Anet.trainMode = false;  
            Dnet.trainMode = false;  
            ts = tic;
            disp(' ')
            disp('Generating adversarial samples... ')
            
            % Generate adversarial samples
            if Anet.cuda
                [advImg, advlabels] = network.adversarialGenerationCUDA(Anet,...
                    thetaA, statesA, normStatA, maxIdxA, imdb, Sx, classObs,...
                    classIdx, NiterTrain);
            else
                [advImg, advlabels] = network.adversarialGeneration(Anet,...
                    thetaA, statesA, normStatA, maxIdxA, imdb, Sx, classObs,...
                    classIdx, NiterTrain);
            end
             
            % Attack network
            disp(' ')
            disp('Attacking network... ')
            if Anet.cuda
                [Pr, sr, er] = network.adversarialAttackRateCUDA(Dnet,...
                    thetaD, statesD, normStatD, maxIdxD, advImg, advlabels,...
                    truelabels, classObs, classIdx, NiterVal);
            else
                [Pr, sr, er] = network.adversarialAttackRate(Dnet, thetaD,...
                    statesD, normStatD, maxIdxD, advImg, advlabels,...
                    truelabels, classObs, classIdx, NiterVal);
            end
            te = toc(ts) / 60;
            
            % Display results
            if Anet.displayMode == 1
                formatSpec = 'Success rate: %3.1f%%, Gobal success rate: %3.1f%%, Total time: %1.0f mins\n';
                fprintf(formatSpec, 100 - 100 * mean(sr), 100 * mean(er), te);
            end
%             if mod(epoch, savedEpoch)==0||epoch==maxEpoch
%                 metric.sr      = sr;
%                 metric.er      = er;
%                 metric.Pr      = Pr;
%                 trainTimeEpoch = te;
%                 task.saveClassificationNet(cdresults, modelName, dataName, theta, normStat, metric, trainTimeEpoch, netInfo, epoch + initEpoch)
%             end
            % Stop
            disp('Done.')
        end
        function adversarialAttack4plot(net, imdb , trainedModelDir)
            % Initialization          
            NiterTrain = round(imdb.numImages /...
                (net.batchSize * net.repBatchSize));
            NiterVal   = round(imdb.numImages *...
                (net.numClasses - 1) / (net.batchSize * net.repBatchSize));
            imgStat    = reshape(net.imgStat', [1, 1, size(net.imgStat, 2) 2]);
            net.imgStat= imgStat;
            imdb.imgStat = imgStat;
            Sx         = cast(net.Sx, net.dtype);
            if net.gpu
                Sx = gpuArray(Sx);
            end
            
            % Encoder Idx
            addIdx   = reshape(repmat(colon(0, net.ny, ...
                (net.batchSize*net.repBatchSize - 1) * net.ny), [net.numClasses, 1]),...
                [net.numClasses * net.batchSize * net.repBatchSize, 1]);
            classObs = repmat(imdb.classObs, [net.batchSize * net.repBatchSize, 1]);
            classIdx = repmat(imdb.classIdx, ...
                [net.batchSize * net.repBatchSize, 1]) +...
                cast(addIdx, class(imdb.classIdx));
            
            % Initialize network
            [net, states, maxIdx, netInfo] = network.initialization(net); 
            if isempty(trainedModelDir)                  
                theta    = tagi.initializeWeightBias(net); 
                normStat = tagi.createInitNormStat(net); 
            else
                l        = load(trainedModelDir);
                theta    = l.theta;
                normStat = l.normStat;
            end
            % Training
            net.trainMode = false;           
            disp(' ')
            disp('Generating adversarial samples... ')
            
            % Generate adversarial samples
            if net.cuda
                [advImg, advlabels] = network.adversarialGenerationCUDA4plot(net,...
                    theta, states, normStat, maxIdx, imdb, Sx, classObs,...
                    classIdx, NiterTrain);
            else
                error('only works with cuda version')
            end
            img = dp.cvr2img(advImg, net.imgSize); 
            pl.class(img, 10, 10, 'samples', 1) 
            disp('Done.')
        end
        
        % Derivative
        function runDerivatveEvaluation(net, xtrain, ytrain, dytrain,...
                Cdxtrain, xtest, ytest, dytest, Sx0, ytrainTrue)
            % Initialization          
            maxEpoch   = net.maxEpoch;
            
            % Train net
            net.trainMode = true;
            [net, states, maxIdx, netInfo] = network.initialization(net);
            
            theta    = tagi.initializeWeightBias(net);
            normStat = tagi.createInitNormStat(net);     
            
            % Test net
            netT              = net;
            netT.trainMode    = false;
            netT.batchSize    = 1;
            netT.repBatchSize = 1;
            [netT, statesT, maxIdxT] = network.initialization(netT); 
            normStatT = tagi.createInitNormStat(netT); 
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Training
            stop  = 0;
            epoch = 0;
            xopt  = 1;%normrnd(0, 1);
            Sxopt = Sx0;
            Sx    = Sxopt;%*ones(size(xtrain), 'like', xtrain);
            while ~stop
                epoch = epoch + 1;
                if epoch >=1
                    idxtrain = randperm(size(ytrain, 1));
                    ytrain   = ytrain(idxtrain, :);
                    xtrain   = xtrain(idxtrain, :);
                    dytrain  = dytrain(idxtrain, :);
                    Cdxtrain = Cdxtrain(idxtrain, :);
                    Sx       = Sx(idxtrain, :);
                    ytrainTrue = ytrainTrue(idxtrain, :);
                end 
                [theta, normStat, yptrain, Syptrain, dyptrain, Sdyptrain,...
                    Cdypxtrain] = network.batchDerivativeCheck(net, theta,...
                    normStat, states, maxIdx, xtrain, Sx, ytrain, 1);
                if epoch >= maxEpoch; break;end
            end
            export_plot=0;
            figure('Position', [0 0 450 250]);
            [xtrain, idxS] = sort(xtrain);
            pl.regression(xtrain, dyptrain(idxS), Sdyptrain(idxS), 'red', 'red', 3)
%             plot(xtrain, dyptrain(idxS), 'ok')
            hold on
            pl.regression(xtrain, dytrain(idxS),...
                9 * (2 * Sx0.^2 + 4 * Sx0 .* xtrain.^2), 'black', 'green', 3)
%             plot(xtrain, dytrain(idxS), 'r')
            xlabel('x')
            ylabel('1st derivative')            
%             legend('prediction', ' ','truth')
            
            set(gcf,'Color',[1 1 1])
            opts=['scaled y ticks = false,',...
                'scaled x ticks = false,',...
                'x label style={font=\huge},',...
                'y label style={font=\huge},',...
                'mark size=5,',...
                'legend style={font=\huge}',...
                ];
            if export_plot==1
                matlab2tikz('figurehandle',gcf,'filename',[ 'saved_figures/' 'derivative_1D' '.tex'] ,'standalone', true,'showInfo', false,...
                    'floatFormat','%.5g','extraTikzpictureOptions','font=\huge','extraaxisoptions',opts);
            end
            
            figure('Position', [0 0 450 250]);
            plot(xtrain, Cdypxtrain(idxS), 'r')
            hold on
            plot(xtrain, Cdxtrain(idxS), 'k')
            xlabel('x')
            ylabel('Cdx')
            legend('prediction', 'truth')
            set(gcf,'Color',[1 1 1])
            opts=['scaled y ticks = false,',...
                'scaled x ticks = false,',...
                'x label style={font=\huge},',...
                'y label style={font=\huge},',...
                'mark size=5,',...
                'legend style={font=\huge}',...
                ];
            if export_plot==1
                matlab2tikz('figurehandle',gcf,'filename',[ 'saved_figures/' 'covariance_derv_x_1D' '.tex'] ,'standalone', true,'showInfo', false,...
                    'floatFormat','%.5g','extraTikzpictureOptions','font=\huge','extraaxisoptions',opts);
            end
            
            figure('Position', [0 0 450 250]);
            pl.regression(xtrain, yptrain(idxS), Syptrain(idxS), 'red', 'red', 3)
            hold on
            plot(xtrain, ytrainTrue(idxS), 'k')
            hold off
            set(gca, 'xtick', [-2, -1, 0, 1, 2],...
                'ytick', [-2, -1, 0, 1, 2]);
            set(gcf,'Color',[1 1 1])
            opts=['scaled y ticks = false,',...
                'scaled x ticks = false,',...
                'x label style={font=\huge},',...
                'y label style={font=\huge},',...
                'mark size=5,',...
                'legend style={font=\huge}',...
                ];
            if export_plot==1
                matlab2tikz('figurehandle',gcf,'filename',[ 'saved_figures/' 'approx_fun_1D' '.tex'] ,'standalone', true,'showInfo', false,...
                    'floatFormat','%.5g','extraTikzpictureOptions','font=\huge','extraaxisoptions',opts);
            end
        end    
        function [xopt] = runOptimization(net, xtrain, ytrain, xtest, ytest,...
                mxtrain, sxtrain, x0, Sx0)
            % Initialization          
            maxEpoch = net.maxEpoch;
            
            % Train net
            net.trainMode = true;
            [net, states, maxIdx] = network.initialization(net);
            
            theta    = tagi.initializeWeightBias(net);
            normStat = tagi.createInitNormStat(net);            
            
            % Test net
            netT              = net;
            netT.trainMode    = false;
            netT.batchSize    = 1;
            netT.repBatchSize = 1;
            [netT, statesT, maxIdxT] = network.initialization(netT); 
            normStatT = tagi.createInitNormStat(netT); 
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Training
            stop      = 0;
            epoch     = 0;
            xopt      = x0;
            Sxopt     = Sx0;
            Sxtrain   = zeros(size(xtrain), 'like', xtrain);
            lastLayer = 1;
            dlayer    = 1;
            tic
            while ~stop
                epoch = epoch + 1;
                if epoch >=1
                    idxtrain = randperm(size(ytrain, 1));
                    ytrain   = ytrain(idxtrain, :);
                    xtrain   = xtrain(idxtrain, :);
                    Sxtrain  = Sxtrain(idxtrain, :);
                end  
                if net.cuda
                    [theta, normStat, yptrain, Syptrain, ~, ~, ~, xopt,...
                        Sxopt] = network.optimizationCUDA(net, theta,...
                        normStat, states, maxIdx, netT, normStatT, statesT,...
                        maxIdxT, xtrain, Sxtrain, ytrain, lastLayer, dlayer,...
                        xopt, Sxopt);
                else
                    [theta, normStat, yptrain, Syptrain, ~, ~, ~, xopt,...
                        Sxopt] = network.optimization(net, theta, normStat,...
                        states, maxIdx, netT, normStatT, statesT, maxIdxT,...
                        xtrain, Sxtrain, ytrain, lastLayer, dlayer, xopt, Sxopt);
                end
                if epoch >= maxEpoch; break;end
            end
            toc
            disp('#################')
            disp('Optimal point:')
            fprintf('% 1.3f +- %1.3f\n', ...
                [xopt(1:net.nx) .* sxtrain + mxtrain, sqrt((sxtrain.^2) ...
                .* Sxopt(1:net.nx))]')
            disp(' ')
            figure  
            [xtrain, idxS] = sort(xtrain(:, 1));
            if net.nx==1
                pl.regression(xtrain, yptrain(idxS), Syptrain(idxS),...
                    'black', 'green', 1)
                plot(xtrain, ytrain(idxS), '--o')
                hold off
            else
                % More than 1 dimmesion
                pl.regression([1:size(xtrain, 1)]', yptrain(idxS),...
                    Syptrain(idxS), 'black', 'green', 1)
                plot([1:size(xtrain, 1)]', ytrain(idxS), '--o')
            hold off
            end
        end
        
        % Classification
        function runClassification(net, trainImdb , valImdb , ...
                trainedModelDir, initModelDir, initEpoch)
            % Initialization          
            cdresults  = net.cd;
            modelName  = net.modelName;
            dataName   = net.dataName;
            savedEpoch = net.savedEpoch;
            maxEpoch   = net.maxEpoch;
            NiterTrain = round(trainImdb.numImages / ...
                (net.batchSize*net.repBatchSize));
            NiterVal   = round(valImdb.numImages /...
                (net.batchSize*net.repBatchSize));
            imgStat    = reshape(net.imgStat', [1, 1, size(net.imgStat, 2) 2]);
            trainImdb.imgStat = imgStat;
            valImdb.imgStat   = imgStat;
            
            % Encoder Idx
            addIdx        = reshape(repmat(colon(0, net.ny, ...
                (net.batchSize*net.repBatchSize-1)*net.ny), ...
                [net.numClasses, 1]), ...
                [net.numClasses * net.batchSize * net.repBatchSize, 1]);
            trainClassObs = repmat(trainImdb.classObs, ...
                [net.batchSize*net.repBatchSize, 1]);
            trainClassIdx = repmat(trainImdb.classIdx, ...
                [net.batchSize*net.repBatchSize, 1]) + ...
                cast(addIdx, class(trainImdb.classIdx));
            
            if isempty(initModelDir)
                [net, states, maxIdx, netInfo] = network.initialization(net); 
            else
                linit   = load(initModelDir);
                net     = linit.net;
                states  = linit.states;
                maxIdx  = linit.maxIdx;
                netInfo = linit.netInfo;
            end
            if isempty(trainedModelDir)                  
                theta = tagi.initializeWeightBias(net); 
                normStat = tagi.createInitNormStat(net); 
            else
                l        = load(trainedModelDir);
                theta    = l.theta;
                normStat = l.normStat;
            end
            % Training
            if net.displayMode
                disp(' ')
                disp('Training... ')
            end
            net.trainMode = true;
            erTrain  = zeros(trainImdb.numImages, net.maxEpoch, net.dtype);
            PnTrain  = zeros(trainImdb.numImages, net.numClasses, ...
                net.maxEpoch, net.dtype);
            erTest   = zeros(valImdb.numImages, net.maxEpoch, net.dtype);
            PnTest   = zeros(valImdb.numImages, net.numClasses, ...
                net.maxEpoch, net.dtype);

            run       = true;
            epoch     = 0;
            trainTime = 0;
            timeTest  = 0;
            while run
                ts = tic;
                epoch = epoch + 1;
                if epoch > 1
                    net.sv = net.sv*net.svDecayFactor;  
                    if net.sv < net.svmin; net.sv = net.svmin; end
                    net.normMomentum = net.normMomentumRef;
                end
                disp('############')
                disp(['Epoch#' num2str(epoch) '/' num2str(maxEpoch)])
                
                % Learning rate schedule
                if any(net.learningRateSchedule == epoch)
                    net.sv = net.scheduledSv(net.learningRateSchedule==epoch);
                end
                
                % Training   
                net.trainMode  = true;
                if net.cuda
                    [theta, normStat,~,~,...
                        sv] = network.classificationCUDA(net, theta, ...
                        normStat, states, maxIdx, trainImdb , trainClassObs,...
                        trainClassIdx, NiterTrain);
                else
                    [theta, normStat,~,~,...
                        sv] = network.classification(net, theta, normStat,...
                        states, maxIdx, trainImdb , trainClassObs,...
                        trainClassIdx, NiterTrain);
                end
                net.sv = sv;
                trainTime = trainTime + toc(ts);
                timeRem   = trainTime / epoch * (net.maxEpoch - epoch) / 60;
                
                % Testing 
                disp('Testing... ')
                tt                = tic;
                net.errorRateEval = 1;
                net.trainMode     = false;
                if net.cuda
                    [~, ~, PnTest(:,:,epoch),...
                        erTest(:, epoch)] = network.classificationCUDA(net,...
                        theta, normStat, states, maxIdx, valImdb,...
                        trainClassObs, trainClassIdx, NiterVal);                      
                else
                    [~, ~, PnTest(:,:,epoch), ...
                        erTest(:, epoch)] = network.classification(net,...
                        theta, normStat, states, maxIdx, valImdb , ...
                        trainClassObs, trainClassIdx, NiterVal); 
                end
                timeTest    = timeTest + toc(tt);
                timeTestRem = timeTest / epoch * (net.maxEpoch - epoch) / 60;
                              
                if net.displayMode == 1
                    formatSpec = 'Error rate: %3.1f%%, Time left: %1.0f mins\n';
                    fprintf(formatSpec, 100 * mean(erTest(:, epoch)),...
                        timeRem + timeTestRem);
                end
                if mod(epoch, savedEpoch) == 0 || epoch == maxEpoch
                    metric.erTest  = erTest;
                    metric.PnTest  = PnTest;
                    metric.erTrain = erTrain;
                    metric.PnTrain = PnTrain;
                    trainTimeEpoch = [trainTime, timeTest];
                    task.saveClassificationNet(cdresults, modelName,...
                        dataName, theta, normStat, metric, trainTimeEpoch,...
                        netInfo, epoch+initEpoch)
                end
                % Stop
                if epoch >= maxEpoch; break; end
            end
            disp('Done.')
        end
        function runDistributionTest(net, valImdb , trainedModelDir,...
                initModelDir)
            % Initialization          
            NiterVal   = round(valImdb.numImages/(net.batchSize*net.repBatchSize));
            labels     = single(imdb.digitlabels);
            imgStat    = reshape(net.imgStat', [1, 1, size(net.imgStat, 2) 2]);
            trainImdb.imgStat = imgStat;
            valImdb.imgStat   = imgStat;
            
            % Encoder Idx
            addIdx        = reshape(repmat(colon(0, net.ny,...
                (net.batchSize * net.repBatchSize - 1) * net.ny),...
                [net.numClasses, 1]), [net.numClasses *...
                net.batchSize * net.repBatchSize, 1]);
            trainClassObs = repmat(trainImdb.classObs,...
                [net.batchSize * net.repBatchSize, 1]);
            trainClassIdx = repmat(trainImdb.classIdx, ...
                [net.batchSize * net.repBatchSize, 1]) +...
                cast(addIdx, class(trainImdb.classIdx));
            
            % Load initial model (pre-calculate indices)
            if isempty(initModelDir)
                [net, states, maxIdx] = network.initialization(net); 
            else
                linit   = load(initModelDir);
                net     = linit.net;
                states  = linit.states;
                maxIdx  = linit.maxIdx;
            end
            
            % Load trained model
            if isempty(trainedModelDir)                  
                theta = tagi.initializeWeightBias(net); 
                normStat = tagi.createInitNormStat(net); 
            else
                l        = load(trainedModelDir);
                theta    = l.theta;
                normStat = l.normStat;
            end
            net.trainMode = false;
            net.errorRateEval = true;
            
            % Testing
            disp('Testing... ')           
            [~, ~, PnTest, erTest] = network.classification_V2(net, theta,...
                normStat, states, maxIdx, valImdb, trainClassObs,...
                trainClassIdx, NiterVal);
            
            % Plot
            pl.plotClassProb(gather(PnTest), gather(labels))
            
            % Display results
            formatSpec = 'Error rate: %3.1f%%';
            fprintf(formatSpec, 100 * mean(erTest(:, epoch)));
            disp('Done.')
        end
        
        % Regression
        function runRegressionFullCov(net, xtrain, Sxtrain, ytrain, xtest,...
                Sxtest, ytest)
            % Initialization          
            cdresults  = net.cd;
            modelName  = net.modelName;
            dataName   = net.dataName;
            saveModel  = net.saveModel;
            maxEpoch   = net.maxEpoch;
            svinit     = net.sv;
            
            % Train net
            net.trainMode = true;
            [net, states, maxIdx, netInfo] = network.initialization(net);
            normStat = tagi.createInitNormStat(net);
            
            % Test net
            netT              = net;
            netT.trainMode    = false;
            netT.batchSize    = 1;
            netT.repBatchSize = 1;
            [netT, statesT, maxIdxT] = network.initialization(netT); 
            normStatT = tagi.createInitNormStat(netT); 
            
            % Initalize weights and bias
            theta    = tagi.initializeWeightBias(net);            
            net.sv   = svinit;
  
            %%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Training
            stop  = 0;
            epoch = 0;
            tic
            while ~stop
                if epoch >1
                    idxtrain = randperm(size(ytrain, 1));
                    ytrain   = ytrain(idxtrain, :);
                    xtrain   = xtrain(idxtrain, :);
                    Sxtrain  = Sxtrain(idxtrain, :);
                    net.sv = 0.85*net.sv';
                    if net.sv<0.01
                        net.sv=0.01;
                    end
                end
                epoch = epoch + 1;
                [theta, normStat, mytrain, Sytrain,...
                    sv] = network.regressionFullCov(net, theta, normStat,...
                    states, maxIdx, xtrain, Sxtrain, ytrain);
                net.sv = sv;
                if epoch >= maxEpoch; break;end
            end
            toc
            Sytrain = Sytrain + net.sv.^2;
            figure('Position', [0 0 450 250]);
            [xtrain, idxS] = sort(xtrain);
            pl.regression(xtrain, mytrain(idxS), Sytrain(idxS), 'red', 'red', 1)
            hold on
            plot(xtrain, ytrain(idxS), 'k')
            hold off
            title('Training set')
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Testing
            [~, ~, mytest, Sytest] = network.regressionFullCov(netT, theta,...
                normStatT, statesT, maxIdxT, xtest, Sxtest, []);
            Sytest = Sytest + net.sv.^2;
            figure('Position', [0 0 450 250]);
            [xtest, idxS] = sort(xtest);
            pl.regression(xtest, mytest(idxS), Sytest(idxS), 'red', 'red', 1)
            hold on
            plot(xtest, ytest(idxS), 'k')
            hold off
            title('Test set')
        end
        function runRegressionDiag(net, xtrain, Sxtrain, ytrain, xtest,...
                Sxtest, ytest)
            % Initialization          
            cdresults  = net.cd;
            modelName  = net.modelName;
            dataName   = net.dataName;
            saveModel  = net.saveModel;
            maxEpoch   = net.maxEpoch;
            svinit     = net.sv;
            
            % Train net
            net.trainMode = 1;
            [net, states, maxIdx, netInfo] = network.initialization(net);
            normStat = tagi.createInitNormStat(net);
            
            % Test net
            netT              = net;
            netT.trainMode    = 0;
            netT.batchSize    = 1;
            netT.repBatchSize = 1;
            [netT, statesT, maxIdxT] = network.initialization(netT); 
            normStatT = tagi.createInitNormStat(netT); 
            
            % Initalize weights and bias
            theta    = tagi.initializeWeightBias(net);            
            net.sv   = svinit;
  
            %%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Training
            stop  = 0;
            epoch = 0;
            tic
            while ~stop
                if epoch >1
                    idxtrain = randperm(size(ytrain, 1));
                    ytrain   = ytrain(idxtrain, :);
                    xtrain   = xtrain(idxtrain, :);
                    Sxtrain  = Sxtrain(idxtrain, :);
                    net.sv = 0.95*net.sv';
                    if net.sv<0.01
                        net.sv=0.01;
                    end
                end
                epoch = epoch + 1;
                [theta, normStat, mytrain, Sytrain,...
                    sv] = network.regression(net, theta, normStat, states,...
                    maxIdx, xtrain, ytrain);
                net.sv = sv;
                if epoch >= maxEpoch; break;end
            end
            toc
            Sytrain = Sytrain + net.sv.^2;
            figure('Position', [0 0 450 250]);
            [xtrain, idxS] = sort(xtrain);
            pl.regression(xtrain, mytrain(idxS), Sytrain(idxS), 'red', 'red', 1)
            hold on
            plot(xtrain, ytrain(idxS), 'k')
            hold off
            title('Training set')
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Testing
            [~, ~, mytest, Sytest] = network.regression(netT, theta, ...
                normStatT, statesT, maxIdxT, xtest, []);
            Sytest = Sytest + net.sv.^2;
            figure('Position', [0 0 450 250]);
            [xtest, idxS] = sort(xtest);
            pl.regression(xtest, mytest(idxS), Sytest(idxS), 'red', 'red', 1)
            hold on
            plot(xtest, ytest(idxS), 'k')
            hold off
            title('Test set')
        end
        
        % Autoencoder
        function runAE(netE, netD, x, trainIdx, trainedModelDir)
            % Initialization
            cd         = netE.cd;
            modelName  = netE.modelName;
            dataName   = netE.dataName;
            savedEpoch = netE.savedEpoch;
            maxEpoch   = netE.maxEpoch;
            Niter      = round(size(x, 4)/(netE.batchSize*netE.repBatchSize));
            
            netE.trainMode = 1;
            netD.trainMode = 1;
            [netE, statesE, maxIdxE, netEinfo] = network.initialization(netE);
            [netD, statesD, maxIdxD, netDinfo] = network.initialization(netD);
            if isempty(trainedModelDir)
                thetaE    = tagi.initializeWeightBias(netE);
                normStatE = tagi.createInitNormStat(netE);
                thetaD    = tagi.initializeWeightBias(netD);
                normStatD = tagi.createInitNormStat(netD);
            else
                l         = load(trainedModelDir);
                thetaE    = l.thetaE;
                normStatE = l.normStatE;
                thetaD    = l.thetaD;
                normStatD = l.normStatD;
            end
            % Data
            xtrain = dp.selectData(x, [], [], [], trainIdx);
            % Training
            if netE.displayMode == 1
                disp(' ')
                disp('Training... ')
            end
            stop      = 0;
            epoch     = 0;
            trainTime = 0;
            while ~stop
                ts    = tic;
                epoch = epoch +1;
                disp('   ############')
                disp(['   Epoch #' num2str(epoch) '/' num2str(netE.maxEpoch)])
                if epoch > 1
                    idxtrain = randperm(size(xtrain, 4));
                    xtrain   = xtrain(:,:,:,idxtrain);
                    netE.sv  = netE.sv*netE.svDecayFactor;
                    netD.sv  = netD.sv*netD.svDecayFactor;
                    if netE.sv < netE.svmin
                        netE.sv = netE.svmin;
                    end
                    if netD.sv < netD.svmin
                        netD.sv = netD.svmin;
                    end
                end
                [thetaE, thetaD, normStatE, normStatD] = network.AE(netE,...
                    thetaE, normStatE, statesE, maxIdxE, netD, thetaD,...
                    normStatD, statesD, maxIdxD, xtrain, Niter);
                trainTime = trainTime + toc(ts);
                timeRem   = trainTime / epoch * (netE.maxEpoch - epoch) / 60;
                
                % Save results after E epochs
                if mod(epoch, savedEpoch) == 0 || epoch == maxEpoch
                    task.saveAEnet(cd, modelName, dataName, thetaE, thetaD,...
                        normStatE, normStatD, netEinfo, netDinfo, epoch);
                end
                if epoch >= maxEpoch
                    stop = 1;
                    disp(' ')
                    disp('Done.')
                end
                if stop ~= 1
                    disp(['   Remaining time: ' sprintf('%0.2f',timeRem) ' mins']);
                end
            end
            % Output
            netE.theta    = thetaE;
            netE.normStat = normStatE;
            netD.theta    = thetaD;
            netD.normStat = normStatD;
        end
        function runAE_V2(net, x, y, trainIdx, trainedModelDir)
            % Initialization          
            cdresults  = net.cd;
            modelName  = net.modelName;
            dataName   = net.dataName;
            savedEpoch = net.savedEpoch;
            maxEpoch   = net.maxEpoch;
            
            net.trainMode = 1;
            [net, states, maxIdx, netInfo] = network.initialization(net); 
            if isempty(trainedModelDir)                  
                theta    = tagi.initializeWeightBias(net); 
                normStat = tagi.createInitNormStat(net); 
            else
                l        = load(trainedModelDir);
                theta    = l.theta;
                normStat = l.normStat;
            end
            % Data
            [xtrain, ytrain, trainLabels,...
                trainEncoderIdx] = dp.selectData(x, y, net.labels,...
                net.encoderIdx, trainIdx); 
            clear x y
            % Training
            net.trainMode = 1;

            stop      = 0;
            epoch     = 0;
            trainTime = 0;
            timeTest  = 0;
            disp(' ')
            disp('Training... ')
            while ~stop
                ts = tic;
                epoch = epoch + 1;
                if epoch>1
                    idxtrain        = randperm(size(ytrain, 1));
                    ytrain          = ytrain(idxtrain, :);
                    xtrain          = xtrain(:,:,:,idxtrain);
                    trainLabels     = trainLabels(idxtrain);
                    trainEncoderIdx = trainEncoderIdx(idxtrain, :);
                end
                disp('   ############')
                disp(['   Epoch #' num2str(epoch) '/' num2str(net.maxEpoch)])
                net.labels     = trainLabels;
                net.encoderIdx = trainEncoderIdx;
                
                [theta, normStat] = network.AE_V2(net, theta, normStat,...
                    states, maxIdx,  xtrain, ytrain);                           
                trainTime = trainTime + toc(ts);
                timeRem   = trainTime / epoch * (net.maxEpoch - epoch) / 60;                
                if epoch >= maxEpoch
                    stop = 1;
                    disp(' ')
                    disp('Done.')
                end
                if stop ~= 1 && net.displayMode == 1
                    disp(['   Remaining time: ' sprintf('%0.2f',timeRem) ' mins']);
                end
                if mod(epoch, savedEpoch)==0||epoch==maxEpoch
                    metric = [];
                    trainTimeEpoch  = [trainTime, timeTest];
                    task.saveClassificationNet(cdresults, modelName, ...
                        dataName, theta, normStat, metric, trainTimeEpoch, netInfo, epoch)
                end
            end
        end        
        
        % GAN
        function runGAN(netD, netG, x, trainIdx, trainedModelDir)
            % Initialization                       
            cdresults = netD.cd;
            modelName = netD.modelName;
            dataName  = netD.dataName;
            savedEpoch = netD.savedEpoch;
            maxEpoch  = netD.maxEpoch;
            
            netD.trainMode = 1;
            netG.trainMode = 1;
            [netD, statesD, maxIdxD, netDinfo] = network.initialization(netD);
            [netG, statesG, maxIdxG, netGinfo] = network.initialization(netG);              
            if isempty(trainedModelDir)                                  
                thetaD    = tagi.initializeWeightBias(netD); 
                normStatD = tagi.createInitNormStat(netD);
                thetaG    = tagi.initializeWeightBias(netG); 
                normStatG = tagi.createInitNormStat(netG);
            else
                l         = load(trainedModelDir);                
                thetaD    = l.thetaD;
                normStatD = l.normStatD;
                thetaG    = l.thetaG;
                normStatG = l.normStatG;
            end
            
            % Data
            xtrain = dp.selectData(x, [], [], [], trainIdx);
            % Training
            if netD.displayMode == 1
                disp(' ')
                disp('Training... ')
            end
            stop      = 0;
            epoch     = 0;
            trainTime = 0;
            while ~stop
                ts    = tic;
                epoch = epoch +1;
                disp('   ############')
                disp(['   Epoch #' num2str(epoch) '/' num2str(netD.maxEpoch)])
                if epoch > 1
                    idxtrain  = randperm(size(xtrain, 4));
                    xtrain    = xtrain(:,:,:,idxtrain);
                    netD.sv = netD.sv*netD.svDecayFactor;
                    if netD.sv < netD.svmin
                        netD.sv = netD.svmin;
                    end
                    netG.sv = netG.sv*netG.svDecayFactor;
                    if netG.sv < netG.svmin
                        netG.sv = netG.svmin;
                    end
                end
                [thetaD, thetaG, normStatD, normStatG] = network.GAN(netD,...
                    thetaD, normStatD, statesD, maxIdxD, netG, thetaG,...
                    normStatG, statesG, maxIdxG, xtrain);
                trainTime = trainTime + toc(ts);
                timeRem   = trainTime / epoch * (netD.maxEpoch - epoch) / 60;
                
                % Save net after E epochs
                if mod(epoch, savedEpoch) == 0 || epoch == maxEpoch
                    task.saveGANnet(cdresults, modelName, dataName, thetaD,...
                        thetaG, normStatD, normStatG, trainTime, netDinfo, netGinfo, epoch)
                end
                
                if epoch >= maxEpoch
                    stop = 1;
                    disp(' ')
                    disp('Done.')
                end
                if stop ~= 1 && netD.displayMode == 1
                    disp(['   Remaining time: ' sprintf('%0.2f',timeRem) ' mins']);
                end
            end
            % Output 
            netD.theta    = thetaD;
            netD.normStat = normStatD;
            netG.theta    = thetaG;
            netG.normStat = normStatG;
        end                        
        function runInfoGAN(netD, netG, netQ, netP, x, trainIdx,...
                trainedModelDir)
            % Initialization 
            cdresults  = netD.cd;
            modelName  = netD.modelName;
            dataName   = netD.dataName;
            savedEpoch = netD.savedEpoch;
            maxEpoch   = netD.maxEpoch;
            Niter      = round(size(x,4)/(netD.batchSize*netD.repBatchSize));
            
            netD.trainMode = true;
            netG.trainMode = true;
            netQ.trainMode = true;
            netP.trainMode = true;
            
            [netD, statesD, maxIdxD, netDinfo] = network.initialization(netD);
            [netG, statesG, maxIdxG, netGinfo] = network.initialization(netG);
            [netQ, statesQ, maxIdxQ, netQinfo] = network.initialization(netQ);
            [netP, statesP, maxIdxP, netPinfo] = network.initialization(netP);
            if isempty(trainedModelDir)                                  
                thetaD    = tagi.initializeWeightBias(netD); 
                normStatD = tagi.createInitNormStat(netD);
                thetaG    = tagi.initializeWeightBias(netG); 
                normStatG = tagi.createInitNormStat(netG);
                thetaQ    = tagi.initializeWeightBias(netQ); 
                normStatQ = tagi.createInitNormStat(netQ);
                thetaP    = tagi.initializeWeightBias(netP); 
                normStatP = tagi.createInitNormStat(netP);
            else
                l         = load(trainedModelDir);                
                thetaD    = l.thetaD;
                normStatD = l.normStatD;
                thetaG    = l.thetaG;
                normStatG = l.normStatG;
                thetaQ    = l.thetaQ;
                normStatQ = l.normStatQ;
                thetaP    = l.thetaP;
                normStatP = l.normStatP;
            end           
            % Data
            xtrain = dp.selectData(x, [], [], [], trainIdx);    
            % Training
            if netD.displayMode == 1
                disp(' ')
                disp('Training... ')
            end
            stop      = 0;
            epoch     = 0;
            trainTime = 0;
            while ~stop
                ts    = tic;
                epoch = epoch +1;
                disp('   ############')
                disp(['   Epoch #' num2str(epoch) '/' num2str(netD.maxEpoch)])
                if epoch > 1
%                     idxtrain = randperm(size(xtrain, 4));
%                     xtrain  = xtrain(:,:,:,idxtrain);
                    netD.sv = netD.sv*netD.svDecayFactor;
                    netG.sv = netG.sv*netG.svDecayFactor;
                    netQ.sv = netQ.sv*netQ.svDecayFactor;
                    netP.sv = netP.sv*netP.svDecayFactor;                   
                    if netD.sv < netD.svmin; netD.sv = netD.svmin;end
                    if netG.sv < netG.svmin; netG.sv = netG.svmin;end
                    if netQ.sv < netQ.svmin; netQ.sv = netQ.svmin;end
                    if netP.sv < netP.svmin; netP.sv = netP.svmin;end
                    netD.normMomentum = netD.normMomentumRef;
                    netP.normMomentum = netP.normMomentumRef;
                    netQ.normMomentum = netQ.normMomentumRef;
                    netG.normMomentum = netG.normMomentumRef;
                end
                [thetaD, thetaG, thetaQ, thetaP, normStatD, normStatG,...
                    normStatQ, normStatP] = network.infoGAN(netD, thetaD,...
                    normStatD, statesD, maxIdxD, netG, thetaG, normStatG,...
                    statesG, maxIdxG, netQ, thetaQ, normStatQ, statesQ,...
                    maxIdxQ, netP, thetaP, normStatP, statesP, maxIdxP,...
                    xtrain,  Niter);
                trainTime = trainTime + toc(ts);
                timeRem   = trainTime/epoch*(netD.maxEpoch-epoch)/60;
                
                if mod(epoch, savedEpoch)==0|| epoch==maxEpoch
                    task.saveinfoGANnet(cdresults, modelName, dataName,...
                        thetaD, thetaG, thetaQ, thetaP, normStatD,...
                        normStatG, normStatQ, normStatP, trainTime,...
                        netDinfo, netGinfo, netQinfo, netPinfo, epoch)
                end
                if epoch >= maxEpoch
                    stop = 1;
                    disp(' ')
                    disp('Done.')
                end
                if stop ~= 1 && netD.displayMode == 1
                    disp(['   Remaining time: ' sprintf('%0.2f',timeRem) ' mins']);
                end
            end
        end                       
        function runACGAN(netD, netG, netQ, netP, x, y, updateIdx, trainIdx, trainedModelDir)
            % Initialization 
            cdresults = netD.cd;
            modelName = netD.modelName;
            dataName  = netD.dataName;
            savedEpoch  = netD.savedEpoch;
            maxEpoch   = netD.maxEpoch;
            
            netD.trainMode = 1;
            netG.trainMode = 1;
            netQ.trainMode = 1;
            netP.trainMode = 1;
            [netD, statesD, maxIdxD, netDinfo] = network.initialization(netD);
            [netG, statesG, maxIdxG, netGinfo] = network.initialization(netG);
            [netQ, statesQ, maxIdxQ, netQinfo] = network.initialization(netQ);
            [netP, statesP, maxIdxP, netPinfo] = network.initialization(netP);
            if isempty(trainedModelDir)                                  
                thetaD    = tagi.initializeWeightBias(netD); 
                normStatD = tagi.createInitNormStat(netD);
                thetaG    = tagi.initializeWeightBias(netG); 
                normStatG = tagi.createInitNormStat(netG);
                thetaQ    = tagi.initializeWeightBias(netQ); 
                normStatQ = tagi.createInitNormStat(netQ);
                thetaP    = tagi.initializeWeightBias(netP); 
                normStatP = tagi.createInitNormStat(netP);
            else
                l         = load(trainedModelDir);                
                thetaD    = l.thetaD;
                normStatD = l.normStatD;
                thetaG    = l.thetaG;
                normStatG = l.normStatG;
                thetaQ    = l.thetaQ;
                normStatQ = l.normStatQ;
                thetaP    = l.thetaP;
                normStatP = l.normStatP;
            end
            % Data
            [xtrain, ytrain, ~, updateIdxtrain] = dp.selectData(x, y,...
                [], updateIdx, trainIdx);
            % Training
            if netD.displayMode == 1
                disp(' ')
                disp('Training... ')
            end
            stop      = 0;
            epoch     = 0;
            trainTime = 0;
            while ~stop
                ts    = tic;
                epoch = epoch +1;
                disp('   ############')
                disp(['   Epoch #' num2str(epoch) '/' num2str(netD.maxEpoch)])
                if epoch > 1
                    idxtrain = randperm(size(xtrain, 4));
                    xtrain   = xtrain(:, :, :, idxtrain);
                    ytrain   = ytrain(idxtrain, :);
                    updateIdxtrain = updateIdxtrain(idxtrain, :);
                    netD.sv  = netD.sv*netD.svDecayFactor;
                    netG.sv  = netG.sv*netG.svDecayFactor;
                    netQ.sv  = netQ.sv*netQ.svDecayFactor;
                    netP.sv  = netP.sv*netP.svDecayFactor;                   
                    if netD.sv < netD.svmin
                        netD.sv = netD.svmin;
                    end
                    if netG.sv < netG.svmin
                        netG.sv = netG.svmin;
                    end
                    if netQ.sv < netQ.svmin
                        netQ.sv = netQ.svmin;
                    end
                    if netP.sv < netP.svmin
                        netP.sv = netP.svmin;
                    end
                end
                [thetaD, thetaG, thetaQ, thetaP, normStatD, normStatG,...
                    normStatQ, normStatP] = network.ACGAN(netD, thetaD,...
                    normStatD, statesD, maxIdxD, netG, thetaG, normStatG,...
                    statesG, maxIdxG, netQ, thetaQ, normStatQ, statesQ,...
                    maxIdxQ, netP, thetaP, normStatP, statesP, maxIdxP, ...
                    xtrain, ytrain, updateIdxtrain);
                trainTime = trainTime + toc(ts);
                timeRem   = trainTime / epoch * (netD.maxEpoch - epoch) / 60;
                % Save after E epochs
                if mod(epoch, savedEpoch)==0||epoch==maxEpoch
                    task.saveinfoGANnet(cdresults, modelName, dataName, ...
                        thetaD, thetaG, thetaQ, thetaP, normStatD, ...
                        normStatG, normStatQ, normStatP, trainTime, ...
                        netDinfo, netGinfo, netQinfo, netPinfo, epoch)
                end
                
                if epoch >= maxEpoch
                    stop = 1;
                    disp(' ')
                    disp('Done.')
                end
                if stop ~= 1 && netD.displayMode == 1
                    disp(['   Remaining time: ' sprintf('%0.2f',timeRem) ' mins']);
                end
            end
            % Output 
            netD.theta    = thetaD;
            netD.normStat = normStatD;
            netG.theta    = thetaG;
            netG.normStat = normStatG;
            netQ.theta    = thetaQ;
            netQ.normStat = normStatQ;
            netP.theta    = thetaP;
            netP.normStat = normStatP;
        end       
                      
        % Save functions
        function saveRegressionNet(cd, modelName, dataName, theta, normStat,...
                metric, trainTime, netInfo, epoch)
            filename = [modelName, '_', 'E', num2str(epoch), '_', dataName];
            folder   = char([cd ,'/results/']);
            save([folder filename], 'theta', 'normStat', 'metric',...
                'trainTime', 'netInfo')
        end
        function saveClassificationNet(cd, modelName, dataName, theta,...
                normStat, metric, trainTime, netInfo, epoch)
            filename = [modelName, '_', 'E', num2str(epoch), '_', dataName];
            folder   = char([cd ,'/results/']);
            save([folder filename], 'theta', 'normStat', 'metric',...
                'trainTime', 'netInfo')
        end
        function saveinfoGANnet(cd, modelName, dataName, thetaD, thetaG,...
                thetaQ, thetaP, normStatD, normStatG, normStatQ, normStatP,...
                trainTime, netDinfo, netGinfo, netQinfo, netPinfo, epoch)
            filename       = [modelName, '_', 'E', num2str(epoch), '_', dataName];
            folder         = char([cd ,'/results/']);
            save([folder filename], 'thetaD',    'thetaG',    'thetaQ',    'thetaP',...
                'normStatD', 'normStatG', 'normStatQ', 'normStatP',...
                'netDinfo',  'netGinfo',  'netQinfo',  'netPinfo', 'trainTime');
        end
        function saveinfoGANnet_V2(cd, modelName, dataName, thetaD, thetaG,...
                thetaQ, thetaQc, thetaP, normStatD, normStatG, normStatQ,...
                normStatQc, normStatP, trainTime, netDinfo, netGinfo, ...
                netQinfo, netQcinfo, netPinfo, epoch)
            filename       = [modelName, '_', 'E', num2str(epoch), '_', dataName];
            folder         = char([cd ,'/results/']);
            save([folder filename], 'thetaD',    'thetaG',    'thetaQ',    'thetaQc', 'thetaP',...
                                    'normStatD', 'normStatG', 'normStatQ', 'normStatQc', 'normStatP',...
                                    'netDinfo',  'netGinfo',  'netQinfo',  'netQcinfo', 'netPinfo', 'trainTime');
        end
        function saveGANnet(cd, modelName, dataName, thetaD, thetaG, ...
                normStatD, normStatG, trainTime, netDinfo, netGinfo, epoch)
            filename  = [modelName, '_', 'E', num2str(epoch), '_', dataName];
            folder    = char([cd ,'/results/']);
            save([folder filename],  'thetaD', 'thetaG', 'normStatD', ...
                'normStatG', 'netDinfo', 'netGinfo', 'trainTime')
        end
        function saveAEnet(cd, modelName, dataName, thetaE, thetaD, ...
                normStatE, normStatD, netEinfo, netDinfo, epoch)
            filename = [modelName, '_', 'E', num2str(epoch), '_', dataName];
            folder   = char([cd ,'/results/']);
            save([folder filename], 'thetaE', 'thetaD', 'normStatE', ...
                'normStatD', 'netDinfo', 'netEinfo')
        end 
        function saveRLDQN1net(cdresults, modelName, dataName, theta, ...
                normStat, stat, netInfo, episode)
            filename = [modelName, '_', 'E', num2str(episode), '_', dataName];
            folder   = char([cdresults, '/results/']);
            save([folder filename], 'theta', 'normStat', 'stat', 'netInfo');
        end
        function saveRL1net(cdresults, modelName, dataName, theta, normStat,...
                stat, netInfo, episode)
            filename = [modelName, '_', 'E', num2str(episode), '_', dataName];
            folder   = char([cdresults, '/results/']);
            save([folder filename], 'theta', 'normStat', 'stat', 'netInfo')
        end
        function saveRL2net(cdresults, modelName, dataName, thetaP, thetaT,...
                normStatP, normStatT, stat, netPinfo, netTinfo, episode)
            filename = [modelName, '_', 'E', num2str(episode), '_', dataName];
            folder   = char([cdresults, '/results/']);
            save([folder filename], 'thetaP', 'normStatP', 'thetaT',...
                'normStatT', 'stat', 'netPinfo', 'netTinfo')
        end
        function saveRLDQN2net(cdresults, modelName, dataName, thetaQ,...
                normStatQ, netQinfo, thetaA, normStatA, netAinfo, stat, episode)
            filename = [modelName, '_', 'E', num2str(episode), '_', dataName];
            folder   = char([cdresults, '/results/']);
            save([folder filename], 'thetaQ', 'normStatQ', 'netQinfo',...
                'thetaA', 'normStatA', 'netAinfo', 'stat');
        end
        
    end
end