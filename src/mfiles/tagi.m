%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         tagi
% Authors:      Luong-Ha Nguyen & James-A. Goulet
% Created:      November 03, 2019
% Updated:      August 26, 2021
% Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
% Copyright (c) 2021 Luong-Ha Nguyen & James-A. Goulet. All rights reserved. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: Tractable Approximate Gaussian Inference (TAGI). 
% This function contains the core of TAGI that includes
% - forward uncertain propagation
% - backward update i.e., inference.
% Further details for each steps, please check out this following paper
% Goulet, Nguyen, and Amiri, (2020): Tractable Approximate Gaussian
% Inference for Bayesian Neural Networks. 
% These two steps are implemented for different types of neural netwworks
% such as FNNs, CNNs, pooling, normalization etc.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 classdef tagi
    methods(Static) 
        % Feedforward
        function [states, normStat, maxIdx, mda, Sda] = feedForwardPass(net,...
                theta, normStat, states, maxIdx)
            % Initialization
            [mw, Sw, mb, Sb, mwx, Swx,...
                mbx, Sbx] = tagi.extractParameters(theta);
            [mz, Sz, ma, Sa, J, mdxs, Sdxs,...
                mxs, Sxs] = tagi.extractStates(states);
            [mra, Sra] = tagi.extractNormStat(normStat);
            numLayers  = length(net.nodes);
            actFunIdx  = net.actFunIdx;
            actBound   = net.actBound;
            layer      = net.layer;
            imgW       = cast(net.imgW, net.dtype);
            imgH       = cast(net.imgH, net.dtype);
            filter     = cast(net.filter, net.dtype);
            kernelSize = cast(net.kernelSize, net.dtype);
            B          = cast(net.batchSize, net.dtype);
            rB         = cast(net.repBatchSize, net.dtype);
            nodes      = cast(net.nodes, net.dtype);
            epsilon    = net.epsilon;
            mhat       = cell(numLayers, 1);
            Shat       = cell(numLayers, 1);
            numParamsPerlayer_2 = net.numParamsPerlayer_2;
            smlIdx     = net.similarIdx;
            
            % Derivative
            mda    = cell(numLayers, 1);
            Sda    = cell(numLayers, 1);
            mda{1} = ones(size(mz{1}), 'like', mz{1});
            Sda{1} = zeros(size(Sz{1}), 'like', Sz{1});
            
            % Hidden Layers
            for j = 2:numLayers
                idxw = (numParamsPerlayer_2(1, j-1)+1):numParamsPerlayer_2(1, j);
                idxb = (numParamsPerlayer_2(2, j-1)+1):numParamsPerlayer_2(2, j);
                % Max pooling
                if layer(j) == net.layerEncoder.mp
                    maPool = normrnd(gather(ma{j-1}), sqrt(abs(gather(Sa{j-1}))));
                    if net.padding(j-1) ~= 0
                        maPool = vertcat(maPool,...
                            -Inf * ones(1, size(maPool, 2), 'like', maPool));
                    end
                    maPool(Sa{j-1}<=0) = -Inf;
                    [mz{j}, Sz{j}, maxIdx{j}] = tagi.mpMeanVar(mz{j}, Sz{j},...
                        maPool, ma{j-1}, Sa{j-1}, net.idxPooling{j-1},...
                        maxIdx{j}, rB, net.gpu);
                    
                    % Average pooling
                elseif layer(j) == net.layerEncoder.ap
                    [mz{j}, Sz{j}] = tagi.apMeanVar(mz{j}, Sz{j}, ma{j-1},...
                        Sa{j-1}, net.idxPooling{j-1}, net.padding(j-1), rB);
                    
                    % Normalization
                elseif layer(j) == net.layerEncoder.ln...
                        || layer(j) == net.layerEncoder.bn
                    
                    if net.trainMode
                        [mhat{j-1}, Shat{j-1}] = tagi.pMeanVar(ma{j-1},...
                            Sa{j-1}, nodes(j-1), imgW(j-1), imgH(j-1),...
                            filter(j-1), B, rB, layer(j-1), layer(j),...
                            net.layerEncoder);
                        % Running average for mean and variance
                        mra{j-1} = net.normMomentum*mra{j-1} +...
                            (1 - net.normMomentum) * mhat{j-1};
                        Sra{j-1} = net.normMomentum*Sra{j-1} +...
                            (1 - net.normMomentum) * Shat{j-1};
                    end
                    mhatD = tagi.distributeNormMeanVar(mra{j-1}, nodes(j-1),...
                        imgW(j-1), imgH(j-1), filter(j-1), B, rB,...
                        layer(j-1), layer(j), net.layerEncoder);
                    ShatD = tagi.distributeNormMeanVar(Sra{j-1}, nodes(j-1),...
                        imgW(j-1), imgH(j-1), filter(j-1), B, rB,...
                        layer(j-1), layer(j), net.layerEncoder);
                    if layer(j-1) == net.layerEncoder.fc
                        [mz{j}, Sz{j}] = tagi.fcNormMeanVar(mz{j}, Sz{j},...
                            mw(idxw), Sw(idxw), mb(idxb), Sb(idxb), ma{j-1},...
                            Sa{j-1}, mhatD, ShatD, epsilon, B, rB, net.gpu);
                    elseif layer(j-1) == net.layerEncoder.conv...
                            || layer(j-1) == net.layerEncoder.tconv
                        [mz{j}, Sz{j}] = tagi.convNormMeanVar(mz{j}, Sz{j},...
                            mw(idxw), Sw(idxw), mb(idxb), Sb(idxb), ma{j-1},...
                            Sa{j-1}, mhatD, ShatD, epsilon, imgH(j-1),...
                            imgH(j-1), filter(j-1), B, rB, net.gpu);
                    end
                    
                    % Convolutional
                elseif layer(j) == net.layerEncoder.conv
                    if B==1&&rB==1
                        [mz{j}, Sz{j}] = tagi.convMeanVarB1(mw(idxw),...
                            Sw(idxw), mb(idxb), Sb(idxb), ma{j-1}, Sa{j-1},...
                            net.idxFmwa(smlIdx(j-1), :), kernelSize(j-1),...
                            filter(j-1), imgW(j), imgH(j), filter(j),...
                            net.padding(j-1), net.gpu);
                    else
                        [mz{j}, Sz{j}] = tagi.convMeanVar(mz{j}, Sz{j},...
                            mw(idxw), Sw(idxw), mb(idxb), Sb(idxb), ma{j-1},...
                            Sa{j-1}, net.idxFmwa(smlIdx(j-1), :),...
                            kernelSize(j-1), filter(j-1), imgW(j), imgH(j),...
                            filter(j), B, rB, net.padding(j-1), net.gpu);
                    end
                    % Transposed convolutional
                elseif layer(j) == net.layerEncoder.tconv
                    [mz{j}, Sz{j}] = tagi.tconvMeanVar(mz{j}, Sz{j},...
                        mw(idxw), Sw(idxw), mb(idxb), Sb(idxb), ma{j-1},...
                        Sa{j-1}, net.idxFmwa(j-1, :), imgW(j), imgH(j),...
                        filter(j), B, rB, net.gpu);
                    
                    % Full-connected
                elseif layer(j) == net.layerEncoder.fc
                    [mz{j}, Sz{j}] = tagi.fcMeanVar(mz{j}, Sz{j}, mw(idxw),...
                        Sw(idxw), mb(idxb), Sb(idxb), ma{j-1}, Sa{j-1},...
                        nodes(j-1), nodes(j), B, rB, net.gpu);
                end
                
                % Shortcut connection for residual networks
                if net.xsc(j)~=0 && (net.filter(net.xsc(j))~=net.filter(j)...
                        ||net.imgW(net.xsc(j))~=net.imgW(j))
                    idxXsc = net.xsc(j);
                    idxwx = (numParamsPerlayer_2(3, idxXsc)+1):numParamsPerlayer_2(3, idxXsc+1);
                    idxbx = (numParamsPerlayer_2(4, idxXsc)+1):numParamsPerlayer_2(4, idxXsc+1);
                    [mxs{j}, Sxs{j}] = tagi.convMeanVar(mxs{j}, Sxs{j},...
                        mwx(idxwx), Swx(idxwx), mbx(idxbx), Sbx(idxbx),...
                        ma{idxXsc}, Sa{idxXsc}, net.idxFmwaXsc(idxXsc, :),...
                        1, filter(idxXsc), imgW(j), imgH(j), filter(j), B,...
                        rB, net.paddingXsc(idxXsc), net.gpu);
                    % Save convolutional hidden state before adding x
                    % shortcut
                    mdxs{j} = mz{j};
                    Sdxs{j} = Sz{j};
                    [mz{j}, Sz{j}] = arrayfun(@twoPlus, mz{j}, Sz{j},...
                        mxs{j}, Sxs{j});
                elseif net.xsc(j)~=0&&(net.filter(net.xsc(j))==net.filter(j)...
                        ||net.imgW(net.xsc(j))~=net.imgW(j))
                    mxs{j}  = mz{net.xsc(j)};
                    Sxs{j}  = Sz{net.xsc(j)};
                    mdxs{j} = mz{j};
                    Sdxs{j} = Sz{j};
                    [mz{j}, Sz{j}] = arrayfun(@twoPlus, mz{j}, Sz{j},...
                        mxs{j}, Sxs{j});
                end
                
                % Activation
                if actFunIdx(j)~=0
                    [ma{j}, Sa{j}, J{j}] = act.meanVar(mz{j}, mz{j}, Sz{j},...
                        actFunIdx(j), actBound(j), B, rB, net.gpu);
                else
                    ma{j} = mz{j};
                    Sa{j} = Sz{j};
                    J{j}  = ones(size(mz{j}), 'like', mz{j});
                end
                
                % Derivative for FC
                if net.collectDev&&actFunIdx(j)~=0
                    [mda{j}, Sda{j}] = act.meanVarDev(mz{j}, Sz{j},...
                        actFunIdx(j), actBound(j));
                end
            end
            normStat = tagi.compressNormStat(mra, Sra);
            states   = tagi.compressStates(mz, Sz, ma, Sa, J, mdxs, Sdxs,...
                mxs, Sxs);
        end      
        function [states, normStat, maxIdx,...
                mda, Sda] = feedForwardPassCUDA(net, theta, normStat,...
                states, maxIdx)
            % Initialization
            [mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx] = tagi.extractParameters(theta);
            [mz, Sz, ma, Sa, J, mdxs, Sdxs, mxs, Sxs] = tagi.extractStates(states);
            [mra, Sra] = tagi.extractNormStat(normStat);
            numLayers  = length(net.nodes);
            actFunIdx  = net.actFunIdx; 
            actBound   = net.actBound;
            layer      = net.layer;
            imgW       = net.imgW;
            imgH       = net.imgH;
            filter     = net.filter;
            kernelSize = net.kernelSize;
            stride     = net.stride;
            B          = net.batchSize;
            rB         = net.repBatchSize;
            nodes      = net.nodes;
            epsilon    = net.epsilon;   
            smlIdx     = net.similarIdx;
            paramUpdateIdx = net.paramUpdateIdx;
            numParamsPerlayer_2 = net.numParamsPerlayer_2;            
            
            % Derivative
            mda    = cell(numLayers, 1);
            Sda    = cell(numLayers, 1);
            mda{1} = ones(size(mz{1}), 'like', mz{1});
            Sda{1} = zeros(size(Sz{1}), 'like', Sz{1});
            % Hidden Layers
            for j = 2:numLayers                  
                % Max pooling
                if layer(j) == net.layerEncoder.mp 
                    maPool = normrnd(gather(ma{j-1}),...
                        sqrt(abs(gather(Sa{j-1}))));
                    if net.padding(j-1) ~= 0
                        maPool = vertcat(maPool,...
                            -Inf * ones(1, size(maPool, 2), 'like', maPool));
                    end
                    maPool(Sa{j-1}<=0) = -Inf;
                    [mz{j}, Sz{j}, maxIdx{j}] = tagi.mpMeanVar(mz{j}, Sz{j},...
                        maPool, ma{j-1}, Sa{j-1}, net.idxPooling{j-1},...
                        maxIdx{j}, rB, net.gpu);
                    
                % Average pooling     
                elseif layer(j) == net.layerEncoder.ap 
                    if kernelSize(j-1) == stride(j-1)...
                            || (kernelSize(j-1) == imgW(j-1)...
                            && stride(j-1) == 1)
                        overlap = 0; 
                    else
                        overlap = 1;
                    end
                    [mz{j},...
                        Sz{j}] = apHiddenStateForwardPass4matlab(ma{j-1},...
                        Sa{j-1}, net.idxPooling{j-1}, imgW(j), imgH(j),...
                        filter(j), imgW(j-1), imgH(j-1), filter(j-1),...
                        kernelSize(j-1), B, overlap); 
                    
                % Normalization     
                elseif layer(j) == net.layerEncoder.ln ...
                        || layer(j) == net.layerEncoder.bn   
                    if net.trainMode
                        % Running average for mean and variance  
                        [mra{j-1}, Sra{j-1}] = normMeanVar4matlab(ma{j-1},...
                            Sa{j-1}, mra{j-1}, Sra{j-1}, nodes(j-1),...
                            imgW(j-1), imgH(j-1), filter(j-1), B,...
                            layer(j-1), layer(j), net.layerEncoder.fc,...
                            net.layerEncoder.conv, net.normMomentum);
                    end                     
                    if layer(j-1) == net.layerEncoder.fc %  TO BE COMPLETED
                        idxw = (numParamsPerlayer_2(1, j-1)+1):numParamsPerlayer_2(1, j);
                        idxb = (numParamsPerlayer_2(2, j-1)+1):numParamsPerlayer_2(2, j);
                        mhatD = tagi.distributeNormMeanVar(mra{j-1},...
                            nodes(j-1), imgW(j-1), imgH(j-1), filter(j-1),...
                            B, rB, layer(j-1), layer(j), net.layerEncoder);
                        ShatD = tagi.distributeNormMeanVar(Sra{j-1},...
                            nodes(j-1), imgW(j-1), imgH(j-1), filter(j-1),...
                            B, rB, layer(j-1), layer(j), net.layerEncoder);
                        [mz{j}, Sz{j}] = tagi.fcNormMeanVar(mz{j}, Sz{j},...
                            mw(idxw), Sw(idxw), mb(idxb), Sb(idxb), ma{j-1},...
                            Sa{j-1}, mhatD, ShatD, epsilon, B, rB, net.gpu);
                    elseif layer(j-1) == net.layerEncoder.conv ...
                            || layer(j-1) == net.layerEncoder.tconv   
                        [mz{j},...
                          Sz{j}] = cnnNormHiddenStateForwardPass4matlab(mw,...
                          Sw, mb, Sb, ma{j-1}, Sa{j-1}, mra{j-1}, Sra{j-1},...
                          epsilon, numParamsPerlayer_2(1, j-1),...
                          numParamsPerlayer_2(2, j-1), imgW(j-1), imgH(j-1),...
                          filter(j-1), B, layer(j));
                    end 
                    
                % Convolutional
                elseif layer(j) == net.layerEncoder.conv    
                    [mz{j}, Sz{j}] = convMeanVar4matlab(mw, Sw, mb, Sb,...
                        ma{j-1}, Sa{j-1}, net.idxFmwa{smlIdx(j-1), 2},...
                        numParamsPerlayer_2(1, j-1),...
                        numParamsPerlayer_2(2, j-1), paramUpdateIdx(2, j-1),...
                        imgW(j), imgH(j), filter(j), imgW(j-1), imgH(j-1),...
                        filter(j-1), kernelSize(j-1), B);
                                      
                % Transposed convolutional    
                elseif layer(j) == net.layerEncoder.tconv                     
                    [mz{j}, Sz{j}] = tconvForwardPass4matlab(mw, Sw, mb, Sb,...
                        ma{j-1}, Sa{j-1}, net.idxFmwa{j-1, 1},...
                        net.idxFmwa{j-1, 2}, numParamsPerlayer_2(1, j-1),...
                        numParamsPerlayer_2(2, j-1), imgW(j), imgH(j),...
                        filter(j), imgW(j-1), imgH(j-1), filter(j-1),...
                        kernelSize(j-1), B);
                    
                % Full-connected
                elseif layer(j) == net.layerEncoder.fc
                    [mz{j}, Sz{j}] = fcMeanVar4matlab(mw, Sw, mb, Sb,...
                        ma{j-1}, Sa{j-1}, numParamsPerlayer_2(1, j-1),...
                        numParamsPerlayer_2(2, j-1), nodes(j), nodes(j-1), B);
                end     
                
%                 % Shortcut connection for residual networks 
                if net.xsc(j) ~= 0 && ... 
                       (net.filter(net.xsc(j)) ~= net.filter(j) ...
                       ||net.imgW(net.xsc(j))~=net.imgW(j)) 
                    idxXsc = net.xsc(j); 
                    [mxs{j}, Sxs{j}] = convMeanVar4matlab(mwx, Swx, mbx,...
                        Sbx, ma{idxXsc}, Sa{idxXsc},...
                        net.idxFmwaXsc{idxXsc, 2},...
                        numParamsPerlayer_2(3, idxXsc),...
                        numParamsPerlayer_2(4, idxXsc),...
                        paramUpdateIdx(4, idxXsc), imgW(j), imgH(j),...
                        filter(j), imgW(idxXsc), imgH(idxXsc),...
                        filter(idxXsc), 1, B);
                    % Save convolutional hidden state before adding x
                    % shortcut
                    mdxs{j} = mz{j};
                    Sdxs{j} = Sz{j};
                    [mz{j}, Sz{j}] = arrayfun(@twoPlus, mz{j}, Sz{j},...
                        mxs{j}, Sxs{j});
                elseif net.xsc(j) ~= 0 && ...
                        (net.filter(net.xsc(j))==net.filter(j) ...
                        || net.imgW(net.xsc(j))~=net.imgW(j))
                    mxs{j}  = mz{net.xsc(j)};
                    Sxs{j}  = Sz{net.xsc(j)};
                    mdxs{j} = mz{j};
                    Sdxs{j} = Sz{j};
                    [mz{j}, Sz{j}] = arrayfun(@twoPlus, mz{j}, Sz{j},...
                        mxs{j}, Sxs{j});
                end
                
                % Activation
                [ma{j}, Sa{j}, J{j}] = activation4matlab(mz{j}, Sz{j},...
                    net.lreluRate, actFunIdx(j));
                              
                % Derivative for FC
                if net.collectDev&&actFunIdx(j)~=0
                    [mda{j}, Sda{j}] = act.meanVarDev(mz{j}, Sz{j},...
                        actFunIdx(j), actBound(j));  
                end
            end 
            normStat = tagi.compressNormStat(mra, Sra);
            states   = tagi.compressStates(mz, Sz, ma, Sa, J, mdxs, Sdxs,...
                mxs, Sxs);
        end
              
        % Inference 
        function [deltaM, deltaS, deltaMx, deltaSx,...
                deltaMz0, deltaSz0, sv] = hiddenStateBackwardPass(net,...
                theta, normStat, states, y, Sy, udIdx, maxIdx)
            % Initialization
            [mw, ~, ~, ~, mwx] = tagi.extractParameters(theta);
            [mz, Sz, ma, Sa, J, mdxs, Sdxs,...
                ~, Sxs] = tagi.extractStates(states);
            [~, Sra] = tagi.extractNormStat(normStat);
            numLayers  = length(net.nodes);
            imgW       = cast(net.imgW, net.dtype);
            imgH       = cast(net.imgH, net.dtype);
            filter     = cast(net.filter, net.dtype);
            kernelSize = cast(net.kernelSize, net.dtype);
            stride     = cast(net.stride, net.dtype);
            B          = cast(net.batchSize, net.dtype);
            rB         = cast(net.repBatchSize, net.dtype);
            nodes      = cast(net.nodes, net.dtype);
            epsilon    = net.epsilon;
            layer      = net.layer;
            lHL        = numLayers-1;
            numParamsPerlayer_2 = net.numParamsPerlayer_2;
            smlIdx     = net.similarIdx;
            
            deltaM     = cell(numLayers, 1);
            deltaS     = cell(numLayers, 1);
            deltaMx    = cell(numLayers, 1);
            deltaSx    = cell(numLayers, 1);
            deltaMxs   = cell(numLayers, 1);
            deltaMdxs  = cell(numLayers, 1);
            deltaSxs   = cell(numLayers, 1);
            deltaSdxs  = cell(numLayers, 1);                      
            if net.lastLayerUpdate
                if net.learnSv == 0
                    % Update hidden states for the last hidden layer
                    if isempty(Sy)
                        R = net.sv.^2;
                    else
                        R = net.sv.^2 + Sy;
                    end                   
                    if isempty(udIdx)
                        Szv = Sa{end} + R;
                        [deltaMz,...
                         deltaSz] = tagi.fowardHiddenStateUpdate(ma{lHL+1},...
                         Szv, J{lHL+1}.*Sz{lHL+1}, y, net.gpu);
                    else
                        mzf = ma{end}(udIdx);
                        Szf = J{lHL+1}(udIdx) .* Sz{lHL+1}(udIdx);
                        ys  = y;
                        Szv = Sa{end}(udIdx) + R;
                        deltaMz = zeros(size(mz{lHL+1}), 'like', mz{lHL+1});
                        deltaSz = zeros(size(Sz{lHL+1}), 'like', Sz{lHL+1});
                        [deltaMz(udIdx),...
                         deltaSz(udIdx)] = tagi.fowardHiddenStateUpdate(mzf,...
                         Szv, Szf, ys, net.gpu);
                    end
                elseif net.learnSv==1                   
                    if strcmp(net.task, 'regression') ...
                            && strcmp(net.noiseType, 'hete')
                        [mla, mv2a] = tagi.detachMeanVar(ma{end}, net.nl,...
                            net.nv2, B, rB);
                        [Sla, Sv2a] = tagi.detachMeanVar(Sa{end}, net.nl,...
                            net.nv2, B, rB);
                        [Slz, ~]  = tagi.detachMeanVar(Sz{end}, net.nl,...
                            net.nv2, B, rB);
                        [Jl, Jv2] = tagi.detachMeanVar(J{end}, net.nl,...
                            net.nv2, B, rB);
                        % Activate log(\sigma_v2)
                        [mv2a, Sv2a, Cv2a] = act.expFun(mv2a, Sv2a, net.gpu);
                        
                        [deltaMlz, deltaSlz, deltaMv2z,...
                            deltaSv2z] = tagi.noiseUpdate4regression(Slz,...
                            mla, Sla, Jl, Jv2, mv2a, Sv2a, Cv2a, y, net.sv,...
                            net.gpu);
                        deltaMz = tagi.attachMeanVar(deltaMlz, deltaMv2z,...
                            net.nl, net.nv2, B, rB);
                        deltaSz = tagi.attachMeanVar(deltaSlz, deltaSv2z,...
                            net.nl, net.nv2, B, rB);
                    elseif strcmp(net.task, 'regression') ...
                            && strcmp(net.noiseType, 'homo')
                        mv2a = net.sv(1);
                        Sv2a = net.sv(2);
                        mla  = ma{end};
                        Slz  = Sz{end};
                        Sla  = Sa{end};
                        Jl   = J{end};  
                        [deltaMz, deltaSz, deltaMv2z,...
                         deltaSv2z] = tagi.homoNoiseUpdate4regression(Slz,...
                         mla, Sla, Jl, mv2a, Sv2a, y, net.gpu);
                        net.sv(1) = net.sv(1) + sum(deltaMv2z, 1);
                        net.sv(2) = net.sv(2) + sum(deltaSv2z, 1);                       
                    elseif strcmp(net.task, 'classification')
                        [mla, mv2a] = tagi.detachMeanVar(ma{end}, net.nl,...
                            net.nv2, B, rB);
                        [Sla, Sv2a] = tagi.detachMeanVar(Sa{end}, net.nl,...
                            net.nv2, B, rB);
                        [Slz, ~]  = tagi.detachMeanVar(Sz{end}, net.nl,...
                            net.nv2, B, rB);
                        [Jl, Jv2] = tagi.detachMeanVar(J{end}, net.nl,...
                            net.nv2, B, rB);
                        % Activate log(\sigma_v2)
                        [mv2a, Sv2a, Cv2a] = act.expFun(mv2a, Sv2a, net.gpu);
                        
                        deltaMlz  = zeros(size(mla), 'like', mla);
                        deltaSlz  = zeros(size(mla), 'like', mla);
                        deltaMv2z = zeros(size(mla), 'like', mla);
                        deltaSv2z = zeros(size(mla), 'like', mla);
                        [deltaMlz(udIdx), deltaSlz(udIdx), deltaMv2z(udIdx),...
                         deltaSv2z(udIdx)] = tagi.noiseUpdate4classification_V2(Slz,...
                         mla, Sla, Jl, Jv2, mv2a, Sv2a, Cv2a, y, net.sv,...
                         udIdx, net.gpu);
                        deltaMz = tagi.attachMeanVar(deltaMlz, deltaMv2z,...
                            net.nl, net.nv2, B, rB);
                        deltaSz = tagi.attachMeanVar(deltaSlz, deltaSv2z,...
                            net.nl, net.nv2, B, rB);
                    end                    
                end
            else
                deltaMz = y;
                deltaSz = Sy;
            end
            sv = net.sv;
            for k = (numLayers-1) : -1 : 1
                if kernelSize(k) == stride(k) ...
                        || (kernelSize(k) == imgW(k) ...
                        && stride(k)==1)
                    overlap = 0;
                else
                    overlap = 1; 
                end
                if isempty(mdxs{k+1}); nSz = Sz{k+1}; else; nSz = Sdxs{k+1}; end
                if isempty(mdxs{k}); cSz = Sz{k}; else; cSz = Sdxs{k}; end
                
                cSxs = Sxs{k};
                idxw = (numParamsPerlayer_2(1, k)+1):numParamsPerlayer_2(1, k+1);
                %Shortcut connection for residual network
                if net.xsc(k+1)~=0 ...
                        && (net.filter(net.xsc(k+1)) ~= net.filter(k+1) ...
                        || net.imgW(net.xsc(k+1)) ~= net.imgH(k+1))
                    [deltaMx{k+1},...
                        deltaSx{k+1}] = tagi.inovationVector(Sxs{k+1},...
                        deltaMzx, deltaSzx, net.gpu);
                    idxXsc = net.xsc(k+1);  
                    idxwx = (numParamsPerlayer_2(3, idxXsc)+1):numParamsPerlayer_2(3, idxXsc+1);
                    if idxXsc>1                                 
                        [deltaMxs{idxXsc}, deltaSxs{idxXsc}, deltaMdxs{idxXsc},...
                         deltaSdxs{idxXsc}] = tagi.xshortDelta(deltaMx{k+1},...
                         deltaSx{k+1}, Sxs{idxXsc}, Sdxs{idxXsc}, J{idxXsc},...
                         mwx(idxwx), net.idxSzzUdXsc{idxXsc},...
                         net.idxFCzwaXsc(idxXsc, :), filter(idxXsc), B, rB,...
                         size(net.idxFCzwaXsc{idxXsc, 2}, 1), net.gpu); 
                    end                   
                elseif net.xsc(k+1)~=0 ...
                        && (net.filter(net.xsc(k+1)) == net.filter(k+1) ...
                        ||net.imgW(net.xsc(k+1)) == net.imgH(k+1))
                    [deltaMx{k+1},...
                        deltaSx{k+1}] = tagi.inovationVector(Sxs{k+1},...
                        deltaMzx, deltaSzx, net.gpu);
                    idxXsc = net.xsc(k+1);
                    if idxXsc > 1 && ~isempty(Sxs{idxXsc})                      
                        [deltaMxs{idxXsc}, deltaSxs{idxXsc}, deltaMdxs{idxXsc},...
                         deltaSdxs{idxXsc}] = tagi.xshortDelta(deltaMx{k+1},...
                         deltaSx{k+1}, Sxs{idxXsc}, Sdxs{idxXsc}, J{idxXsc},...
                         [],  [], [], [], [], rB, [], net.gpu);
                    elseif idxXsc > 1 ...
                            && isempty(Sdxs{idxXsc}) ...
                            && isempty(Sxs{idxXsc}) % First shortcut
                        [~, ~, deltaMdxs{idxXsc},...
                         deltaSdxs{idxXsc}] = tagi.xshortDelta(deltaMx{k+1},...
                         deltaSx{k+1}, [], Sz{idxXsc}, J{idxXsc}, [], [],...
                         [], [], [], rB, [], net.gpu);
                    end
                end   
                
                % Innovation vector
                [deltaM{k+1}, deltaS{k+1}] = tagi.inovationVector(nSz,...
                    deltaMz, deltaSz, net.gpu);
                
                % Max pooling 
                if layer(k+1) == net.layerEncoder.mp       
                    [deltaMz, deltaSz, deltaMzx,...
                        deltaSzx] = tagi.mpHiddenStateBackwardPass(cSz, cSxs,...
                        J{k}, deltaM{k+1}, deltaS{k+1}, maxIdx{k+1}, rB,...
                        overlap, net.gpu);
                    
                % Average pooling     
                elseif layer(k+1) == net.layerEncoder.ap 
                    [deltaMz, deltaSz, deltaMzx,...
                        deltaSzx] = tagi.agHiddenStateBackwardPass(cSz, cSxs,...
                        J{k}, size(net.idxPooling{k}, 2), deltaM{k+1},...
                        deltaS{k+1}, net.idxSzzUd{k}, imgW(k+1), imgH(k+1),...
                        filter(k+1), kernelSize(k), B, rB, overlap, net.gpu);
                    
                % Convolutional     
                elseif layer(k+1) == net.layerEncoder.conv 
                    if k > 1||net.convariateEstm
                        if B == 1 && rB == 1
                            [deltaMz, deltaSz, deltaMzx,...
                             deltaSzx] = tagi.convHiddenStateBackwardPassB1(cSz,...
                             cSxs, J{k}, mw(idxw), deltaM{k+1}, deltaS{k+1},...
                            net.idxSzzUd{smlIdx(k)},...
                            net.idxFCzwa(smlIdx(k), :),...
                            imgW(k), imgH(k), filter(k), net.gpu);
                        else
                            [deltaMz, deltaSz, deltaMzx,...
                             deltaSzx] = tagi.convHiddenStateBackwardPass(cSz,...
                             cSxs, J{k}, mw(idxw), deltaM{k+1}, deltaS{k+1},...
                             net.idxSzzUd{smlIdx(k)},...
                             net.idxFCzwa(smlIdx(k), :), imgW(k), imgH(k),...
                             filter(k), B, rB, net.gpu);
                        end                       
                    end
                    
                % Transposed convolutional
                elseif layer(k+1) == net.layerEncoder.tconv 
                    if k > 1 || net.convariateEstm
                        [deltaMz, deltaSz, deltaMzx,...
                         deltaSzx] = tagi.tconvHiddenStateBackwardPass(cSz,...
                         cSxs, J{k}, mw(idxw), deltaM{k+1}, deltaS{k+1},...
                         net.idxSzzUd{k}, net.idxFCzwa(k, :), imgW(k),...
                         imgH(k), filter(k), B, rB, net.gpu);                       
                    end
                    
                % Normalization     
                elseif layer(k+1) == net.layerEncoder.ln ...
                        || layer(k+1) == net.layerEncoder.bn                     
                    if k > 1 || net.convariateEstm
                        Shat = tagi.distributeNormMeanVar(Sra{k}, nodes(k),...
                            imgW(k), imgH(k), filter(k), B, rB, layer(k),...
                            layer(k+1), net.layerEncoder);
                        [deltaMz, deltaSz, deltaMzx,...
                            deltaSzx] = tagi.normHiddenStateBackwardPass(cSz,...
                            cSxs, J{k}, mw(idxw), Shat, epsilon, deltaM{k+1},...
                            deltaS{k+1}, imgW(k), imgH(k), filter(k), B, rB,...
                            layer(k), net.layerEncoder, net.gpu);
                    end 
                    
                % Full-connected     
                elseif  layer(k+1) == net.layerEncoder.fc
                    if k > 1||net.convariateEstm
                        [deltaMz, deltaSz, deltaMzx,...
                         deltaSzx] = tagi.fcHiddenStateBackwardPass(cSz,...
                         cSxs, J{k}, mw(idxw), deltaM{k+1}, deltaS{k+1},...
                         nodes(k), nodes(k+1), B, rB, net.gpu);
                    end                  
                end
                
                % Update hidden states from shortcut
                if ~isempty(deltaMxs{k}) && ~isempty(deltaMdxs{k})
                    [deltaMzx, deltaSzx, deltaMz,...
                        deltaSz] = arrayfun(@fourPlus, deltaMzx, deltaSzx,...
                        deltaMz, deltaSz, deltaMxs{k}, deltaSxs{k},...
                        deltaMdxs{k}, deltaSdxs{k});
                elseif ~isempty(deltaMdxs{k}) && isempty(deltaMxs{k})
                    [deltaMz, deltaSz] = arrayfun(@twoPlus, deltaMz, deltaSz,...
                        deltaMdxs{k}, deltaSdxs{k});
                end
            end
            deltaMz0 = deltaMz;
            deltaSz0 = deltaSz;        
        end
        function [deltaMzopt,...
                deltaSzopt] = optimizationHiddenStateBackwardPass(net,...
                theta, normStat, states, y, Sy, maxIdx, lastLayer, dlayer)
            % Initialization
            [mw, ~, ~, ~, mwx] = tagi.extractParameters(theta);
            [~, Sz, ~, ~, J, mdxs, Sdxs,...
                ~, Sxs] = tagi.extractStates(states);
            [~, Sra]   = tagi.extractNormStat(normStat);
            numLayers  = length(net.nodes);
            imgW       = cast(net.imgW, net.dtype);
            imgH       = cast(net.imgH, net.dtype);
            filter     = cast(net.filter, net.dtype);
            kernelSize = cast(net.kernelSize, net.dtype);
            stride     = cast(net.stride, net.dtype);
            B          = cast(net.batchSize, net.dtype);
            rB         = cast(net.repBatchSize, net.dtype);
            nodes      = cast(net.nodes, net.dtype);
            epsilon    = net.epsilon;
            layer      = net.layer;          
            smlIdx     = net.similarIdx;
            numParamsPerlayer_2 = net.numParamsPerlayer_2;
            
            deltaM    = cell(numLayers, 1);
            deltaS    = cell(numLayers, 1);
            deltaMx   = cell(numLayers, 1);
            deltaSx   = cell(numLayers, 1);
            deltaMxs  = cell(numLayers, 1);
            deltaMdxs = cell(numLayers, 1);
            deltaSxs  = cell(numLayers, 1);
            deltaSdxs = cell(numLayers, 1);  
            
            deltaMz = y;
            deltaSz = Sy;
            for k = lastLayer - 1 : -1 : dlayer
                if kernelSize(k)==stride(k)...
                        || (kernelSize(k)==imgW(k)...
                        && stride(k)==1)
                    overlap = 0;
                else
                    overlap = 1;
                end
                if isempty(mdxs{k+1}); nSz = Sz{k+1}; else; nSz = Sdxs{k+1}; end
                if isempty(mdxs{k}); cSz = Sz{k}; else; cSz = Sdxs{k}; end
                
                cSxs = Sxs{k};
                idxw = (numParamsPerlayer_2(1, k)+1):numParamsPerlayer_2(1, k+1);   
                
                %Shortcut connection for residual network
                if net.xsc(k+1)~=0 && ...
                        (net.filter(net.xsc(k+1))~=net.filter(k+1) ...
                        || net.imgW(net.xsc(k+1))~=net.imgH(k+1))
                    [deltaMx{k+1},...
                        deltaSx{k+1}] = tagi.inovationVector(Sxs{k+1},...
                        deltaMzx, deltaSzx, net.gpu);
                    idxXsc = net.xsc(k+1);  
                    idxwx = (numParamsPerlayer_2(3, idxXsc)+1):numParamsPerlayer_2(3, idxXsc+1);
                    if idxXsc > 1                                 
                        [deltaMxs{idxXsc}, deltaSxs{idxXsc}, deltaMdxs{idxXsc},...
                         deltaSdxs{idxXsc}] = tagi.xshortDelta(deltaMx{k+1},...
                         deltaSx{k+1}, Sxs{idxXsc}, Sdxs{idxXsc}, J{idxXsc},...
                         mwx(idxwx), net.idxSzzUdXsc{idxXsc},...
                         net.idxFCzwaXsc(idxXsc, :), filter(idxXsc), B, rB,...
                         size(net.idxFCzwaXsc{idxXsc, 2}, 1), net.gpu); 
                    end                   
                elseif net.xsc(k+1) ~= 0 ...
                        && (net.filter(net.xsc(k+1)) == net.filter(k+1) ...
                        ||net.imgW(net.xsc(k+1)) == net.imgH(k+1))
                    [deltaMx{k+1},...
                        deltaSx{k+1}] = tagi.inovationVector(Sxs{k+1},...
                        deltaMzx, deltaSzx, net.gpu);
                    idxXsc = net.xsc(k+1);
                    if idxXsc > 1 && ~isempty(Sxs{idxXsc})                      
                        [deltaMxs{idxXsc}, deltaSxs{idxXsc}, deltaMdxs{idxXsc},...
                         deltaSdxs{idxXsc}] = tagi.xshortDelta(deltaMx{k+1},...
                         deltaSx{k+1}, Sxs{idxXsc}, Sdxs{idxXsc}, J{idxXsc},...
                         [], [], [], [], [], rB, [], net.gpu);
                    elseif idxXsc > 1 && isempty(Sdxs{idxXsc}) ...
                            && isempty(Sxs{idxXsc}) % First shortcut
                        [~, ~, deltaMdxs{idxXsc},...
                         deltaSdxs{idxXsc}] = tagi.xshortDelta(deltaMx{k+1},...
                         deltaSx{k+1}, [], Sz{idxXsc}, J{idxXsc}, [], [],...
                         [], [], [], rB, [], net.gpu);
                    end
                end   
                
                % Innovation vector
                [deltaM{k+1}, deltaS{k+1}] = tagi.inovationVector(nSz,...
                    deltaMz, deltaSz, net.gpu);
                
                % Max pooling 
                if layer(k+1) == net.layerEncoder.mp       
                    [deltaMz, deltaSz, deltaMzx,...
                        deltaSzx] = tagi.mpHiddenStateBackwardPass(cSz,...
                        cSxs, J{k}, deltaM{k+1}, deltaS{k+1}, maxIdx{k+1},...
                        rB, overlap, net.gpu);
                    
                % Average pooling     
                elseif layer(k+1) == net.layerEncoder.ap 
                    [deltaMz, deltaSz, deltaMzx,...
                     deltaSzx] = tagi.agHiddenStateBackwardPass(cSz, cSxs,...
                     J{k}, size(net.idxPooling{k}, 2), deltaM{k+1}, deltaS{k+1},...
                     net.idxSzzUd{k}, imgW(k+1), imgH(k+1), filter(k+1),...
                     kernelSize(k), B, rB, overlap, net.gpu);
                    
                % Convolutional     
                elseif layer(k+1) == net.layerEncoder.conv 
                    if k > 1||net.convariateEstm
                        if B == 1 && rB == 1
                            [deltaMz, deltaSz, deltaMzx,...
                             deltaSzx] = tagi.convHiddenStateBackwardPassB1(cSz,...
                             cSxs, J{k}, mw(idxw), deltaM{k+1}, deltaS{k+1},...
                             net.idxSzzUd{smlIdx(k)},...
                             net.idxFCzwa(smlIdx(k), :), imgW(k), imgH(k),...
                             filter(k), net.gpu);
                        else
                            [deltaMz, deltaSz, deltaMzx,...
                             deltaSzx] = tagi.convHiddenStateBackwardPass(cSz,...
                             cSxs, J{k}, mw(idxw), deltaM{k+1}, deltaS{k+1},...
                             net.idxSzzUd{smlIdx(k)},...
                             net.idxFCzwa(smlIdx(k), :), imgW(k), imgH(k),...
                             filter(k), B, rB, net.gpu);
                        end                       
                    end
                    
                % Transposed convolutional
                elseif layer(k+1) == net.layerEncoder.tconv 
                    if k > 1||net.convariateEstm
                        [deltaMz, deltaSz, deltaMzx,...
                         deltaSzx] = tagi.tconvHiddenStateBackwardPass(cSz,...
                         cSxs, J{k}, mw(idxw), deltaM{k+1}, deltaS{k+1},...
                         net.idxSzzUd{k}, net.idxFCzwa(k, :), imgW(k),...
                         imgH(k), filter(k), B, rB, net.gpu);                       
                    end
                    
                % Normalization     
                elseif layer(k+1) == net.layerEncoder.ln ...
                        || layer(k+1) == net.layerEncoder.bn                     
                    if k > 1||net.convariateEstm
                        Shat = tagi.distributeNormMeanVar(Sra{k}, nodes(k),...
                            imgW(k), imgH(k), filter(k), B, rB, layer(k),...
                            layer(k+1), net.layerEncoder);
                        [deltaMz, deltaSz, deltaMzx,...
                         deltaSzx] = tagi.normHiddenStateBackwardPass(cSz,...
                         cSxs, J{k}, mw(idxw), Shat, epsilon, deltaM{k+1},...
                         deltaS{k+1}, imgW(k), imgH(k), filter(k), B, rB,...
                         layer(k), net.layerEncoder, net.gpu);
                    end 
                    
                % Full-connected     
                elseif  layer(k+1) == net.layerEncoder.fc
                    if k > 1||net.convariateEstm
                        if B==1&&rB==1
                            [deltaMz, deltaSz, deltaMzx,...
                             deltaSzx] = tagi.fcHiddenStateBackwardPassB1(cSz,...
                             cSxs, J{k}, mw(idxw), deltaM{k+1}, deltaS{k+1},...
                             nodes(k), nodes(k+1), net.gpu);
                        else
                            [deltaMz, deltaSz, deltaMzx,...
                                deltaSzx] = tagi.fcHiddenStateBackwardPass(cSz,...
                                cSxs, J{k}, mw(idxw), deltaM{k+1}, deltaS{k+1},...
                                nodes(k), nodes(k+1), B, rB, net.gpu);
                        end                                               
                    end                  
                end
                
                % Update hidden states from shortcut
                if ~isempty(deltaMxs{k})&&~isempty(deltaMdxs{k})
                    [deltaMzx, deltaSzx, deltaMz,...
                        deltaSz] = arrayfun(@fourPlus, deltaMzx, deltaSzx,...
                        deltaMz, deltaSz, deltaMxs{k}, deltaSxs{k},...
                        deltaMdxs{k}, deltaSdxs{k});
                elseif ~isempty(deltaMdxs{k})&&isempty(deltaMxs{k})
                    [deltaMz, deltaSz] = arrayfun(@twoPlus, deltaMz,...
                        deltaSz, deltaMdxs{k}, deltaSdxs{k});
                end                
            end
            deltaMzopt = deltaMz;
            deltaSzopt = deltaSz;        
        end
        function [mdg, Sdg, Cdgz] = derivative(net, theta, normStat, states,...
                mda, Sda, dlayer)
            % Initialization
            [mw, Sw] = tagi.extractParameters(theta);
            [~, Sz, ma, Sa, J] = tagi.extractStates(states);
            [~, Sra]  = tagi.extractNormStat(normStat);
            numLayers = length(net.nodes);
            actFunIdx = net.actFunIdx; 
            B         = cast(net.batchSize, net.dtype);
            rB        = cast(net.repBatchSize, net.dtype);
            nodes     = cast(net.nodes, net.dtype);
            nodes     = [nodes, nodes(end)]; 
            layer     = net.layer;
            numParamsPerlayer_2 = net.numParamsPerlayer_2;
            
            % Derivative 
            mdg  = tagi.createDevCellarray(nodes, numLayers, B, rB, net.dtype, net.gpu); 
            Sdg  = mdg;
            Cdgz = mdg;
            mdge = mdg; 
            if layer(numLayers)==net.layerEncoder.bn
                sl = 2;
            else
                sl =1;
            end
            for k = (numLayers-sl) : -1 : dlayer  
                idxw = (numParamsPerlayer_2(1, k)+1):numParamsPerlayer_2(1, k+1);   
                if  layer(k+1) == net.layerEncoder.fc
                    if net.collectDev && k == numLayers-sl
                        [mdgk, Sdgk]   = tagi.fcMeanVarDnode(mw(idxw),...
                            Sw(idxw), mda{k}, Sda{k}, nodes(k), nodes(k+1),...
                            B);
                        [Caizi, Caozi] = tagi.fcCovaz(J{k+1}, J{k}, Sz{k},...
                            mw(idxw), nodes(k), nodes(k+1), B);
                        [Cdozi, Cdizi] = tagi.fcCovdz(ma{k+1}, ma{k}, Caizi,...
                            Caozi, actFunIdx(k+1), actFunIdx(k), nodes(k),...
                            nodes(k+1), B);
                        mdg{k}  = sum(mdgk, 2);
                        Sdg{k}  = sum(Sdgk, 2);
                        mdge{k} = mdgk;   
                        Cdgzk   = tagi.covdx(0, mw(idxw), 1 + zeros(net.ny * B, 1),...
                            1, ones(net.ny * B, 1), Cdozi, Cdizi, nodes(k),...
                            nodes(k+1), 1, B);
                        Cdgz{k} = sum(Cdgzk, 2);
                    elseif net.collectDev && k < numLayers - sl
                        [mdgk, Sdgk, Cdgzk] = tagi.fcDerivative(mw(idxw),...
                            Sw(idxw), mw(idxwo), J{k+1}, J{k}, ma{k+1},...
                            Sa{k+1}, ma{k}, Sa{k}, Sz{k}, mda{k}, Sda{k},...
                            mdg{k+1}, mdge{k+1}, Sdg{k+1}, mdg{k+2},...
                            actFunIdx(k+1), actFunIdx(k), nodes(k),...
                            nodes(k+1), nodes(k+2), B);
                        mdg{k}  = sum(mdgk, 2);
                        Sdg{k}  = sum(Sdgk, 2);
                        Cdgz{k} = sum(Cdgzk, 2);
                        mdge{k} = mdgk;
                    end 
                elseif layer(k+1) == net.layerEncoder.bn
                        mdg{k}  = repmat(mw(idxw)./sqrt(Sra{k} + 1E-8), [B, rB]);
                        Sdg{k}  = repmat(Sw(idxw)./(Sra{k} + 1E-8), [B, rB]);
                        Cdgz{k} = 0;
                        mdge{k} = mw(idxw)./sqrt(Sra{k} + 1E-8);
                end          
                idxwo = idxw;
            end    
        end
        function deltaTheta = parameterBackwardPass(net, theta, normStat,...
                states, deltaM, deltaS, deltaMx, deltaSx)
            % Initialization
            [mw, Sw, mb, Sb, mwx, Swx,...
                mbx, Sbx] = tagi.extractParameters(theta);
            [~, ~, ma, ~, ~, ~, ~, ~, ~] = tagi.extractStates(states);
            [mra, Sra] = tagi.extractNormStat(normStat);
            numLayers  = length(net.nodes);
            imgW       = cast(net.imgW, net.dtype);
            imgH       = cast(net.imgH, net.dtype);
            filter     = cast(net.filter, net.dtype);
            kernelSize = cast(net.kernelSize, net.dtype);
            B          = cast(net.batchSize, net.dtype);
            rB         = cast(net.repBatchSize, net.dtype);
            nodes      = cast(net.nodes, net.dtype);
            epsilon    = net.epsilon;
            layer      = net.layer;
            numParamsPerlayer_2 = net.numParamsPerlayer_2;
            smlIdx     = net.similarIdx;
            
            deltaMw    = mw;
            deltaSw    = Sw;
            deltaMb    = mb;
            deltaSb    = Sb;
            deltaMwx   = mwx;
            deltaSwx   = Swx;
            deltaMbx   = mbx;
            deltaSbx   = Sbx;
            for k = (numLayers - 1) : -1 : 1
                idxw = (numParamsPerlayer_2(1, k)+1):numParamsPerlayer_2(1, k+1);
                idxb = (numParamsPerlayer_2(2, k)+1):numParamsPerlayer_2(2, k+1);
                %Shortcut connection for residual network
                if net.xsc(k+1)~=0 ...
                        && (net.filter(net.xsc(k+1)) ~= net.filter(k+1) ...
                        ||net.imgW(net.xsc(k+1)) ~= net.imgH(k+1))
                    idxXsc = net.xsc(k+1); 
                    idxwx = (numParamsPerlayer_2(3, idxXsc)+1):numParamsPerlayer_2(3, idxXsc+1);
                    idxbx = (numParamsPerlayer_2(4, idxXsc)+1):numParamsPerlayer_2(4, idxXsc+1);
                    
                    [deltaMwx(idxwx), deltaSwx(idxwx), deltaMbx(idxbx),...
                     deltaSbx(idxbx)] = tagi.convParameterBackwardPass(deltaMwx(idxwx),...
                     deltaSwx(idxwx), deltaMbx(idxbx), deltaSbx(idxbx),...
                     Swx(idxwx), Sbx(idxbx), ma{idxXsc}, deltaMx{k+1},...
                     deltaSx{k+1}, net.idxFmwaXsc(idxXsc, :),...
                     net.paddingXsc(idxXsc), 1, filter(idxXsc), imgW(k+1),...
                     imgH(k+1), filter(k+1), B, rB, net.gpu);
                end 
                
                % Convolutional     
                if layer(k+1) == net.layerEncoder.conv  
                    if B == 1 && rB == 1
                        [deltaMw(idxw), deltaSw(idxw), deltaMb(idxb),...
                         deltaSb(idxb)] = tagi.convParameterBackwardPassB1(Sw(idxw),...
                         Sb(idxb), ma{k}, deltaM{k+1}, deltaS{k+1},...
                         net.idxFmwa(smlIdx(k), :), net.padding(k),...
                         kernelSize(k), filter(k), imgW(k+1), imgH(k+1),...
                         filter(k+1), net.gpu);
                    else
                        [deltaMw(idxw), deltaSw(idxw), deltaMb(idxb),...
                         deltaSb(idxb)] = tagi.convParameterBackwardPass(deltaMw(idxw),...
                         deltaSw(idxw), deltaMb(idxb), deltaSb(idxb),...
                         Sw(idxw), Sb(idxb), ma{k}, deltaM{k+1}, deltaS{k+1},...
                         net.idxFmwa(smlIdx(k), :), net.padding(k),...
                         kernelSize(k), filter(k), imgW(k+1), imgH(k+1),...
                         filter(k+1), B, rB, net.gpu);
                    end                                                         
                % Transposed convolutional
                elseif layer(k+1) == net.layerEncoder.tconv 
                    [deltaMw(idxw), deltaSw(idxw), deltaMb(idxb),...
                     deltaSb(idxb)] = tagi.tconvParameterBackwardPass(deltaMw(idxw),...
                     deltaSw(idxw), deltaMb(idxb), deltaSb(idxb), Sw(idxw),...
                     Sb(idxb), ma{k}, deltaM{k+1}, deltaS{k+1},...
                     net.idxSwzUd{k}, net.idxFCwz(k, :), kernelSize(k),...
                     filter(k), imgW(k+1), imgH(k+1), filter(k+1), B, rB,...
                     net.gpu); 
                    
                % Normalization     
                elseif layer(k+1) == net.layerEncoder.ln ...
                        || layer(k+1) == net.layerEncoder.bn  
                    mhat = tagi.distributeNormMeanVar(mra{k}, nodes(k),...
                        imgW(k), imgH(k), filter(k), B, rB, layer(k), layer(k+1), net.layerEncoder);
                    Shat = tagi.distributeNormMeanVar(Sra{k}, nodes(k),...
                        imgW(k), imgH(k), filter(k), B, rB, layer(k),...
                        layer(k+1), net.layerEncoder);
                    [deltaMw(idxw), deltaSw(idxw), deltaMb(idxb),...
                     deltaSb(idxb)] = tagi.normParameterBackwardPass(deltaMw(idxw),...
                     deltaSw(idxw), deltaMb(idxb), deltaSb(idxb), Sw(idxw),...
                     Sb(idxb), ma{k}, mhat, Shat, epsilon, deltaM{k+1},...
                     deltaS{k+1}, nodes(k), imgW(k), imgH(k), filter(k), B,...
                     rB, layer(k), net.layerEncoder, net.gpu);        
                    
                % Full-connected     
                elseif  layer(k+1) == net.layerEncoder.fc 
                    [deltaMw(idxw), deltaSw(idxw), deltaMb(idxb),...
                     deltaSb(idxb)] = tagi.fcParameterBackwardPass(deltaMw(idxw),...
                     deltaSw(idxw), deltaMb(idxb), deltaSb(idxb), Sw(idxw),...
                     Sb(idxb), ma{k}, deltaM{k+1}, deltaS{k+1}, nodes(k),...
                     nodes(k+1), B, rB, net.gpu);
                end
            end
            deltaTheta = tagi.compressParameters(deltaMw, deltaSw, deltaMb,...
                deltaSb, deltaMwx, deltaSwx, deltaMbx, deltaSbx);           
        end             
        function [deltaM, deltaS, deltaMx, deltaSx,...
                deltaMz0, deltaSz0, sv] = hiddenStateBackwardPassCUDA(net,...
                theta, normStat, states, y, Sy, udIdx, maxIdx)
            % Initialization
            [mw, ~, ~, ~, mwx] = tagi.extractParameters(theta);
            [~, Sz, ma, Sa, J, mdxs, Sdxs,...
                ~, Sxs] = tagi.extractStates(states);
            [~, Sra] = tagi.extractNormStat(normStat);
            numLayers  = length(net.nodes);
            imgW       = net.imgW;
            imgH       = net.imgH;
            filter     = net.filter;
            kernelSize = net.kernelSize;
            stride     = net.stride;
            B          = net.batchSize;
            rB         = net.repBatchSize;
            nodes      = net.nodes;
            epsilon    = net.epsilon;
            layer      = net.layer;
            numParamsPerlayer_2 = net.numParamsPerlayer_2;
            smlIdx     = net.similarIdx;
            
            deltaM     = cell(numLayers, 1);
            deltaS     = cell(numLayers, 1);
            deltaMx    = cell(numLayers, 1);
            deltaSx    = cell(numLayers, 1);
            deltaMxs   = cell(numLayers, 1);
            deltaMdxs  = cell(numLayers, 1);
            deltaSxs   = cell(numLayers, 1);
            deltaSdxs  = cell(numLayers, 1);  
            if net.lastLayerUpdate              
                if net.learnSv == 0
                    % Update hidden states for the last hidden layer
                    if isempty(Sy)
                        R = net.sv.^2;
                    else
                        R = net.sv.^2 + Sy;
                    end                   
                    if isempty(udIdx)
                        Szv = Sa{end} + R;
                        [deltaMz,...
                         deltaSz] = tagi.fowardHiddenStateUpdate(ma{end},...
                         Szv, J{end}.*Sz{end}, y, net.gpu);
                    else
                        mzf = ma{end}(udIdx);
                        Szf = J{end}(udIdx) .* Sz{end}(udIdx);
                        ys  = y;
                        Szv = Sa{end}(udIdx) + R;
                        deltaMz = zeros(size(mz{end}), 'like', mz{end});
                        deltaSz = zeros(size(Sz{end}), 'like', Sz{end});
                        [deltaMz(udIdx),...
                         deltaSz(udIdx)] = tagi.fowardHiddenStateUpdate(mzf,...
                         Szv, Szf, ys, net.gpu);
                    end
%                     % Update hidden states for the last hidden layer
%                     if ~isempty(Sy)
%                         Sy = Sy + net.sv.^2;
%                     else
%                         Sy = (net.sv.^2) * ones(size(Sa{end}), 'like', Sa{end});
%                     end
%                     [deltaMz, deltaSz] = lastLayerUpdate4matlab(ma{end},...
%                         Sa{end}, Sz{end}, J{end}, y, Sy, net.ny, net.nye,...
%                         double(rB * B), net.isUdidx, udIdx);
                elseif net.learnSv == 1                   
                    if strcmp(net.task, 'regression') && strcmp(net.noiseType, 'hete')
                        [mla, mv2a] = tagi.detachMeanVar(ma{end}, net.nl,...
                            net.nv2, B, rB);
                        [Sla, Sv2a] = tagi.detachMeanVar(Sa{end}, net.nl,...
                            net.nv2, B, rB);
                        [Slz, ~]  = tagi.detachMeanVar(Sz{end}, net.nl,...
                            net.nv2, B, rB);
                        [Jl, Jv2] = tagi.detachMeanVar(J{end}, net.nl,...
                            net.nv2, B, rB);
                        % Activate log(\sigma_v2)
                        [mv2a, Sv2a, Cv2a] = act.expFun(mv2a, Sv2a, net.gpu);
                        
                        [deltaMlz, deltaSlz, deltaMv2z,...
                            deltaSv2z] = tagi.noiseUpdate4regression(Slz,...
                            mla, Sla, Jl, Jv2, mv2a, Sv2a, Cv2a, y, net.sv,...
                            net.gpu);
                        deltaMz = tagi.attachMeanVar(deltaMlz, deltaMv2z,...
                            net.nl, net.nv2, B, rB);
                        deltaSz = tagi.attachMeanVar(deltaSlz, deltaSv2z,...
                            net.nl, net.nv2, B, rB);
                    elseif strcmp(net.task, 'regression') && strcmp(net.noiseType, 'homo')
                        mv2a = net.sv(1);
                        Sv2a = net.sv(2);
                        mla  = ma{end};
                        Slz  = Sz{end};
                        Sla  = Sa{end};
                        Jl   = J{end};  
                        [deltaMz, deltaSz, deltaMv2z,...
                         deltaSv2z] = tagi.homoNoiseUpdate4regression(Slz,...
                         mla, Sla, Jl, mv2a, Sv2a, y, net.gpu);
                        net.sv(1) = net.sv(1) + sum(deltaMv2z, 1);
                        net.sv(2) = net.sv(2) + sum(deltaSv2z, 1);                       
                    elseif strcmp(net.task, 'classification')
                        [mla, mv2a] = tagi.detachMeanVar(ma{end}, net.nl,...
                            net.nv2, B, rB);
                        [Sla, Sv2a] = tagi.detachMeanVar(Sa{end}, net.nl,...
                            net.nv2, B, rB);
                        [Slz, ~]  = tagi.detachMeanVar(Sz{end}, net.nl,...
                            net.nv2, B, rB);
                        [Jl, Jv2] = tagi.detachMeanVar(J{end}, net.nl,...
                            net.nv2, B, rB);
                        % Activate log(\sigma_v2)
                        [mv2a, Sv2a, Cv2a] = act.expFun(mv2a, Sv2a, net.gpu);
                        
                        deltaMlz  = zeros(size(mla), 'like', mla);
                        deltaSlz  = zeros(size(mla), 'like', mla);
                        deltaMv2z = zeros(size(mla), 'like', mla);
                        deltaSv2z = zeros(size(mla), 'like', mla);
                        [deltaMlz(udIdx), deltaSlz(udIdx), deltaMv2z(udIdx),...
                         deltaSv2z(udIdx)] = tagi.noiseUpdate4classification_V2(Slz,...
                         mla, Sla, Jl, Jv2, mv2a, Sv2a, Cv2a, y, net.sv,...
                         udIdx, net.gpu);
                        deltaMz = tagi.attachMeanVar(deltaMlz, deltaMv2z,...
                            net.nl, net.nv2, B, rB);
                        deltaSz = tagi.attachMeanVar(deltaSlz, deltaSv2z,...
                            net.nl, net.nv2, B, rB);
                    end                    
                end
            else
                deltaMz = y;
                deltaSz = Sy;
            end
            sv = net.sv;
            if any(net.xsc~=0)
                firstShorcut = net.xsc(find(net.xsc, 1, 'first'));
            else
                firstShorcut = nan;
            end
            idxXscUd = nan;
            for k = (numLayers-1):-1:1
                if kernelSize(k) == stride(k) ...
                        ||(kernelSize(k) == imgW(k) ...
                        && stride(k)==1)
                    overlap = 0;
                else
                    overlap = 1; 
                end
                if net.xsc(k+1)==0; nSz = Sz{k+1}; else; nSz = Sdxs{k+1}; end
                if net.xsc(k)==0; cSz = Sz{k}; else; cSz = Sdxs{k}; end
                
                cSxs = Sxs{k};                  
                %Shortcut connection for residual network
                if net.xsc(k+1)~=0 ...
                        && (net.filter(net.xsc(k+1)) ~= net.filter(k+1) ...
                        ||net.imgW(net.xsc(k+1)) ~= net.imgH(k+1)) ...
                        && net.xsc(k+1)>1
                    [deltaMx{k+1},...
                        deltaSx{k+1}] = inovationVector4matlab(Sxs{k+1},...
                        deltaMzx, deltaSzx);
                    idxXsc = net.xsc(k+1);  
                    [deltaMxs{idxXsc}, deltaSxs{idxXsc}, deltaMdxs{idxXsc},...
                     deltaSdxs{idxXsc}] = tagi.convscHiddenStateBackwardPassMexcuda(mwx,...
                     J{idxXsc}, Sxs{idxXsc}, Sdxs{idxXsc}, deltaMx{k+1},...
                     deltaSx{k+1}, numParamsPerlayer_2(3, idxXsc),...
                     net.idxFCzwaXsc(idxXsc, :), net.idxSzzUdXsc{idxXsc},...
                     imgW(k+1), imgH(k+1), filter(k+1), imgW(idxXsc),...
                     imgH(idxXsc), filter(idxXsc), 1, B);
                    idxXscUd = idxXsc;
                    
                elseif net.xsc(k+1)~=0 ...
                        && (net.filter(net.xsc(k+1)) == net.filter(k+1) ...
                        || net.imgW(net.xsc(k+1)) == net.imgH(k+1)) ...
                        && net.xsc(k+1)>1
                    [deltaMx{k+1},...
                        deltaSx{k+1}] = inovationVector4matlab(Sxs{k+1},...
                        deltaMzx, deltaSzx);
                    idxXsc = net.xsc(k+1);
                    if idxXsc ~= firstShorcut 
                        [deltaMxs{idxXsc},...
                         deltaSxs{idxXsc}]   = scHiddenStateBackwardPass4matlab(Sxs{idxXsc},...
                         J{idxXsc}, deltaMx{k+1}, deltaSx{k+1});
                        [deltaMdxs{idxXsc},...
                         deltaSdxs{idxXsc}] = scHiddenStateBackwardPass4matlab(Sdxs{idxXsc},...
                         J{idxXsc}, deltaMx{k+1}, deltaSx{k+1});
                    else % First shortcut
                        [deltaMdxs{idxXsc},...
                         deltaSdxs{idxXsc}] = scHiddenStateBackwardPass4matlab(Sz{idxXsc},...
                         J{idxXsc}, deltaMx{k+1}, deltaSx{k+1}); 
                    end
                    idxXscUd = idxXsc;
                end   
                
                % Innovation vector
                [deltaM{k+1}, deltaS{k+1}] = inovationVector4matlab(nSz,...
                    deltaMz, deltaSz);
                
                % Max pooling 
                if layer(k+1) == net.layerEncoder.mp       
                    [deltaMz, deltaSz, deltaMzx,...
                     deltaSzx] = tagi.mpHiddenStateBackwardPass(cSz, cSxs,...
                     J{k}, deltaM{k+1}, deltaS{k+1}, maxIdx{k+1}, rB, overlap, net.gpu);
                    
                % Average pooling     
                elseif layer(k+1) == net.layerEncoder.ap 
                    [deltaMz, deltaSz] = apHiddenStateBackwardPass4matlab(cSz,...
                     J{k}, deltaM{k+1}, deltaS{k+1}, net.idxSzzUd{k},...
                     imgW(k+1), imgH(k+1),  filter(k+1), imgW(k), imgH(k),...
                     filter(k), kernelSize(k), B, overlap);
                    if ~isempty(cSxs)
                        [deltaMzx,...
                         deltaSzx] = apHiddenStateBackwardPass4matlab(cSxs,...
                         J{k}, deltaM{k+1}, deltaS{k+1}, net.idxSzzUd{k},...
                         imgW(k+1), imgH(k+1), filter(k+1), imgW(k),...
                         imgH(k), filter(k), kernelSize(k), B, overlap);
                    else
                        deltaMzx = [];
                        deltaSzx = [];
                    end
                    
                % Convolutional     
                elseif layer(k+1) == net.layerEncoder.conv 
                    if k > 1||net.convariateEstm
                        [deltaMz,...
                         deltaSz] = convHiddenStateBackwardPass4matlab_V2(mw,...
                         cSz, J{k}, deltaM{k+1}, deltaS{k+1},...
                         net.idxFCzwa{smlIdx(k), 1}, net.idxSzzUd{smlIdx(k)},...
                         numParamsPerlayer_2(1, k), imgW(k+1), imgH(k+1),...
                         filter(k+1), imgW(k), imgH(k), filter(k),...
                         kernelSize(k), B);
                        if ~isempty(cSxs)
                            [deltaMzx,...
                             deltaSzx] = convHiddenStateBackwardPass4matlab_V2(mw,...
                             cSxs, J{k}, deltaM{k+1}, deltaS{k+1},...
                             net.idxFCzwa{smlIdx(k), 1},...
                             net.idxSzzUd{smlIdx(k)},...
                             numParamsPerlayer_2(1, k), imgW(k+1), imgH(k+1),...
                             filter(k+1), imgW(k), imgH(k), filter(k),...
                             kernelSize(k), B);
                        else
                            deltaMzx = [];
                            deltaSzx = [];
                        end                           
                    end
                    
                % Transposed convolutional
                elseif layer(k+1) == net.layerEncoder.tconv 
                    if k > 1 || net.convariateEstm                       
                        [deltaMz,...
                         deltaSz] = tconvHiddenStateBackwardPass4matlab(mw,...
                         cSz, J{k}, deltaM{k+1}, deltaS{k+1},...
                         net.idxFCzwa{k, 1}, net.idxSzzUd{k},...
                         numParamsPerlayer_2(1, k), imgW(k+1), imgH(k+1),...
                         filter(k+1), imgW(k), imgH(k), filter(k),...
                         kernelSize(k), B);   
                        deltaMzx = [];
                        deltaSzx = [];
                    end
                    
                % Normalization     
                elseif layer(k+1) == net.layerEncoder.ln ...
                        || layer(k+1) == net.layerEncoder.bn  
                    if k > 1 || net.convariateEstm
                        [deltaMz, deltaSz, deltaMzx,...
                         deltaSzx] = tagi.normHiddenStateBackwardPassMexcuda(cSz,...
                         cSxs, J{k}, mw, Sra{k}, deltaM{k+1}, deltaS{k+1},...
                         epsilon, numParamsPerlayer_2(1, k), nodes(k),...
                         imgW(k), imgH(k), filter(k), B, layer(k),...
                         layer(k+1), net.layerEncoder);                    
                    end 
                    
                % Full-connected     
                elseif  layer(k+1) == net.layerEncoder.fc
                    if k > 1||net.convariateEstm
                        [deltaMz,...
                         deltaSz] = fcHiddenStateBackwardPass4matlab(mw,...
                         cSz, J{k}, deltaM{k+1}, deltaS{k+1},...
                         numParamsPerlayer_2(1, k), nodes(k), nodes(k+1), B);
                        deltaMzx = [];
                        deltaSzx = [];                                           
                    end                  
                end
                
                % Update hidden states from shortcut
                if k == idxXscUd && k ~= firstShorcut
                    [deltaMzx, deltaSzx, deltaMz,...
                        deltaSz] = arrayfun(@fourPlus, deltaMzx, deltaSzx,...
                        deltaMz, deltaSz, deltaMxs{k}, deltaSxs{k},...
                        deltaMdxs{k}, deltaSdxs{k});
                    idxXscUd = nan;
                elseif k == idxXscUd && k == firstShorcut
                    [deltaMz, deltaSz] = arrayfun(@twoPlus, deltaMz,...
                        deltaSz, deltaMdxs{k}, deltaSdxs{k});
                    idxXscUd = nan;
                end
            end
            deltaMz0 = deltaMz;
            deltaSz0 = deltaSz;        
        end
        function deltaTheta = parameterBackwardPassCUDA(net, theta,...
                normStat, states, deltaM, deltaS, deltaMx, deltaSx)
            % Initialization
            [mw, Sw, mb, Sb, mwx, Swx,...
                mbx, Sbx] = tagi.extractParameters(theta);
            [~, ~, ma] = tagi.extractStates(states);
            [mra, Sra] = tagi.extractNormStat(normStat);
            numLayers  = length(net.nodes);
            imgW       = net.imgW;
            imgH       = net.imgH;
            filter     = net.filter;
            kernelSize = net.kernelSize;
            B          = net.batchSize;
            rB         = net.repBatchSize;
            nodes      = net.nodes;
            epsilon    = net.epsilon;
            layer      = net.layer;
            smlIdx     = net.similarIdx;
            paramUpdateIdx = net.paramUpdateIdx;
            numParamsPerlayer_2 = net.numParamsPerlayer_2;
            
            deltaMw    = mw;
            deltaSw    = Sw;
            deltaMb    = mb;
            deltaSb    = Sb;
            deltaMwx   = mwx;
            deltaSwx   = Swx;
            deltaMbx   = mbx;
            deltaSbx   = Sbx;
            for k = (numLayers-1):-1:1               
                %Shortcut connection for residual network
                if net.xsc(k+1)~=0 ...
                        && (net.filter(net.xsc(k+1)) ~= net.filter(k+1) ...
                        || net.imgW(net.xsc(k+1)) ~= net.imgH(k+1))
                    idxXsc = net.xsc(k+1); 
                    [deltaMwx, deltaSwx, deltaMbx,...
                     deltaSbx] = convParamBackwardPass4matlab(Swx, ma{idxXsc},...
                     Sbx, deltaMx{k+1}, deltaSx{k+1}, net.idxFmwaXsc{idxXsc, 2},...
                     numParamsPerlayer_2(3, idxXsc),...
                     numParamsPerlayer_2(4, idxXsc),...
                     paramUpdateIdx(4, idxXsc), imgW(k+1), imgH(k+1),...
                     filter(k+1), imgW(idxXsc), imgH(idxXsc), filter(idxXsc),...
                     1, B, deltaMwx, deltaSwx, deltaMbx, deltaSbx);                     
                end 
                                
                % Convolutional     
                if layer(k+1) == net.layerEncoder.conv 
                    [deltaMw, deltaSw, deltaMb,...
                     deltaSb] = convParamBackwardPass4matlab(Sw, ma{k}, Sb,...
                     deltaM{k+1}, deltaS{k+1}, net.idxFmwa{smlIdx(k), 2},...
                     numParamsPerlayer_2(1, k), numParamsPerlayer_2(2, k),...
                     paramUpdateIdx(2, k), imgW(k+1), imgH(k+1), filter(k+1),...
                     imgW(k), imgH(k), filter(k), kernelSize(k), B, deltaMw,...
                     deltaSw, deltaMb, deltaSb);
                    
                % Transposed convolutional
                elseif layer(k+1) == net.layerEncoder.tconv 
                    [deltaMw, deltaSw, deltaMb,...
                     deltaSb] = tconvParamBackwardPass4matlab(Sw, ma{k}, Sb,...
                     deltaM{k+1}, deltaS{k+1}, net.idxFCwz{k, 2},...
                     net.idxSwzUd{k}, numParamsPerlayer_2(1, k),...
                     numParamsPerlayer_2(2, k), imgW(k+1), imgH(k+1),...
                     filter(k+1), imgW(k), imgH(k), filter(k), kernelSize(k),...
                     B, deltaMw, deltaSw, deltaMb, deltaSb);
                    
                % Normalization     
                elseif layer(k+1) == net.layerEncoder.ln ...
                        || layer(k+1) == net.layerEncoder.bn 
                    if layer(k) == net.layerEncoder.conv ...
                            || layer(k) == net.layerEncoder.tcov % Previous layer is convolutional
                        [deltaMw, deltaSw, deltaMb,...
                         deltaSb] = convNormParamBackwardPass4matlab(Sw, Sb,...
                         ma{k}, mra{k}, Sra{k}, deltaM{k+1}, deltaS{k+1},...
                         epsilon, numParamsPerlayer_2(1, k),...
                         numParamsPerlayer_2(2, k), imgW(k), imgH(k),...
                         filter(k), B, layer(k+1), deltaMw, deltaSw,...
                         deltaMb, deltaSb);                      
                    elseif layer(k) == net.layerEncoder.fc % Previous layer is full-connected
                        [deltaMw, deltaSw, deltaMb,...
                         deltaSb] = fcNormParamBackwardPass4matlab(Sw, Sb,...
                         ma{k}, mra{k}, Sra{k}, deltaM{k+1}, deltaS{k+1},...
                         epsilon, numParamsPerlayer_2(1, k),...
                         numParamsPerlayer_2(2, k), nodes(k), B, layer(k+1),...
                         deltaMw, deltaSw, deltaMb, deltaSb);
                    end   
                    
                % Full-connected     
                elseif  layer(k+1) == net.layerEncoder.fc 
                    [deltaMw, deltaSw, deltaMb,...
                     deltaSb] = fcParamBackwardPass4matlab(Sw, ma{k}, Sb,...
                     deltaM{k+1}, deltaS{k+1}, numParamsPerlayer_2(1, k),...
                     numParamsPerlayer_2(2, k),  nodes(k), B, nodes(k+1),...
                     nodes(k+1), B, 1, deltaMw, deltaSw, deltaMb, deltaSb);
                end
            end
            deltaTheta = tagi.compressParameters(deltaMw, deltaSw, deltaMb,...
                deltaSb, deltaMwx, deltaSwx, deltaMbx, deltaSbx);           
        end  
        
        % Pooling layer
        function [mz, Sz, maxIdx] = mpMeanVar(mz, Sz, maS, ma, Sa,...
                idxpooling, maxIdx, rB, gpu)
            n = size(idxpooling, 1);   
            for t = 1:rB
                maSloop = maS(:, t);
                [~, idx] = max(maSloop(idxpooling), [], 2);
                if gpu
                    n   = gpuArray(cast(n, 'int32'));
                    col = gpuArray.colon(1, n);
                    col = col(:);
                    fun = @(x, y, z) (x-1).*y + z;
                    idx = arrayfun(fun, idx, n, col);
                else
                    col = colon(1,n)';
                    idx = (idx-1)*n + col;
                end
                maxIdx(:, t) = idxpooling(idx);
                mz(:, t) = ma(maxIdx(:, t), t);
                Sz(:, t) = Sa(maxIdx(:, t), t);
            end
        end
        function [mz, Sz] = apMeanVar(mz, Sz, ma, Sa, idxPooling, padding,...
                rB)
            n   = size(idxPooling, 2);
            if padding ~= 0
                zeroPad = zeros(1, size(ma, 2), 'like', ma);
                ma = [ma; zeroPad];
                Sa = [Sa; zeroPad];
            end
            for t = 1:rB  
                maloop = ma(:, t);
                Saloop = Sa(:, t);
                mz(:, t) = mean(maloop(idxPooling), 2);
                Sz(:, t) = sum(Saloop(idxPooling), 2)./(n^2);
            end           
        end             
        function [deltaMz, deltaSz,...
                deltaMxs, deltaSxs] = mpHiddenStateBackwardPass(Sz, Sxs, J,...
                deltaM, deltaS, maxIdx, rB, overlap, gpu)
            deltaMz  = Sz;
            deltaSz  = Sz;
            deltaMxs = Sxs;
            deltaSxs = Sxs;
            n = single(size(Sz, 1));
            if gpu
                if isempty(Sxs)
                    for t = 1:rB
                        Czz = bsxfun(@times, J(:, t), Sz(:, t));
                        Czz = Czz(maxIdx(:, t));
                        if overlap == 1
                            [deltaMzloop,...
                             deltaSzloop] = arrayfun(@vectorizedDelta, Czz,...
                             deltaM(:, t), deltaS(:, t));
                            deltaMz(:, t) = accumarray(maxIdx(:, t),...
                             deltaMzloop, [n, 1], @sum);
                            deltaSz(:, t) = accumarray(maxIdx(:, t),...
                             deltaSzloop , [n, 1], @sum);
                        else
                            [deltaMz(maxIdx(:, t), t),...
                              deltaSz(maxIdx(:, t), t)] = arrayfun(@vectorizedDelta,...
                              Czz, deltaM(:, t), deltaS(:, t));
                        end
                    end
                else
                    for t = 1:rB
                        if overlap == 1
                            [deltaMzloop, deltaSzloop, deltaMxsloop,...
                              deltaSxsloop] = arrayfun(@vectorized4delta,...
                              J(maxIdx(:, t), t), Sz(maxIdx(:, t), t),...
                              Sxs(maxIdx(:, t), t), deltaM(:, t), deltaS(:, t));
                            deltaMz(:, t)  = accumarray(maxIdx(:, t),...
                                deltaMzloop, [n, 1], @sum);
                            deltaSz(:, t)  = accumarray(maxIdx(:, t),...
                                deltaSzloop , [n, 1], @sum);
                            deltaMxs(:, t) = accumarray(maxIdx(:, t),...
                                deltaMxsloop, [n, 1], @sum);
                            deltaSxs(:, t) = accumarray(maxIdx(:, t),...
                                deltaSxsloop , [n, 1], @sum);
                        else
                            [deltaMz(maxIdx(:, t), t),...
                              deltaSz(maxIdx(:, t), t),...
                              deltaMxs(maxIdx(:, t), t),...
                              deltaSxs(maxIdx(:, t), t)] = arrayfun(@vectorized4delta,...
                              J(maxIdx(:, t), t), Sz(maxIdx(:, t), t),...
                              Sxs(maxIdx(:, t), t), deltaM(:, t), deltaS(:, t));
                        end
                    end
                end
            else
                if isempty(Sxs)
                    for t = 1:rB
                        Czz = J(:, t).*Sz(:, t);
                        Czz = Czz(maxIdx(:, t));
                        if overlap == 1
                            deltaMzloop   = Czz.*deltaM(:, t);
                            deltaSzloop   = Czz.*deltaS(:, t) .* Czz;
                            deltaMz(:, t) = accumarray(maxIdx(:, t), ...
                                deltaMzloop, [n, 1], @sum);
                            deltaSz(:, t) = accumarray(maxIdx(:, t), ...
                                deltaSzloop , [n, 1], @sum);
                        else
                            deltaMz(maxIdx(:, t), t) = Czz .* deltaM(:, t);
                            deltaSz(maxIdx(:, t), t) = Czz .* deltaS(:, t) .* Czz;
                        end
                    end
                else
                    for t = 1:rB
                        Czz = J(:, t) .* Sz(:, t);
                        Czz = Czz(maxIdx(:, t));
                        Czx = J(:, t) .* Sxs(:, t);
                        Czx = Czx(maxIdx(:, t));
                        if overlap == 1
                            deltaMzloop    = Czz .* deltaM(:, t);
                            deltaSzloop    = Czz .* deltaS(:, t) .* Czz;
                            deltaMxsloop   = Czx .* deltaM(:, t);
                            deltaSxsloop   = Czx .* deltaS(:, t) .* Czx;
                            deltaMz(:, t)  = accumarray(maxIdx(:, t),...
                                deltaMzloop, [n, 1], @sum);
                            deltaSz(:, t)  = accumarray(maxIdx(:, t),...
                                deltaSzloop, [n, 1], @sum);
                            deltaMxs(:, t) = accumarray(maxIdx(:, t),...
                                deltaMxsloop, [n, 1], @sum);
                            deltaSxs(:, t) = accumarray(maxIdx(:, t),...
                                deltaSxsloop, [n, 1], @sum);
                        else
                            deltaMz(maxIdx(:, t), t) = Czz .* deltaM(:, t);
                            deltaSz(maxIdx(:, t), t) = Czz .* deltaS(:, t) .* Czz;
                        end
                    end
                end
            end
        end              
        function [deltaMz, deltaSz,...
                deltaMxs, deltaSxs] = agHiddenStateBackwardPass(Sz, Sxs, J,...
                n, deltaM, deltaS, idx,  wo, ho, fo, ki, B, rB, overlap, gpu)    
            deltaMz  = Sz;
            deltaSz  = Sz;
            deltaMxs = Sxs;
            deltaSxs = Sxs;
            n = cast(n, 'like', Sz);
            if gpu
                if isempty(Sxs)
                    for t = 1:rB
                        if overlap == 0
                            deltaMzloop = reshape(repmat(reshape(repmat(transpose(deltaM(:, t)),...
                                [ki, 1]), [ki * ho, wo * fo * B]), [ki, 1]),...
                                [ho * wo * fo * ki * ki * B, 1]);
                            deltaSzloop = reshape(repmat(reshape(repmat(transpose(deltaS(:, t)),...
                                [ki, 1]), [ki * ho, wo * fo * B]), [ki, 1]),...
                                [ho * wo * fo * ki * ki * B, 1]);
                        else
                            zeroPadding = zeros(1,1,'like',deltaM);
                            deltaMzloop = [deltaM(:, t); zeroPadding];
                            deltaSzloop = [deltaS(:, t); zeroPadding];
                            deltaMzloop = deltaMzloop(idx);
                            deltaSzloop = deltaSzloop(idx);
                        end
                        [deltaMzloop,...
                          deltaSzloop] = arrayfun(@vectorizedDelta_V2,...
                          J(:, t), Sz(:, t)/n, deltaMzloop, deltaSzloop);
                        deltaMz(:, t) = sum(deltaMzloop, 2);
                        deltaSz(:, t) = sum(deltaSzloop, 2);
                    end
                else
                    Czz = bsxfun(@times, J, Sz);
                    Czx = bsxfun(@times, J, Sxs);
                    for t = 1:rB
                        if overlap == 0
                            deltaMloop = reshape(repmat(reshape(repmat(transpose(deltaM(:, t)),...
                                [ki, 1]), [ki * ho, wo * fo * B]), [ki, 1]),...
                                [ho * wo * fo * ki * ki * B, 1]);
                            deltaSloop = reshape(repmat(reshape(repmat(transpose(deltaS(:, t)),...
                                [ki, 1]), [ki * ho, wo * fo *B ]), [ki, 1]),...
                                [ho * wo * fo * ki * ki * B, 1]);
                        else
                            zeroPadding = zeros(1,1,'like',deltaM);
                            deltaMloop = [deltaM(:, t); zeroPadding];
                            deltaSloop = [deltaS(:, t); zeroPadding];
                            deltaMloop = deltaMloop(idx);
                            deltaSloop = deltaSloop(idx);
                        end
                        [deltaMzloop, deltaSzloop, deltaMxsloop,...
                         deltaSxsloop] = arrayfun(@vectorized4delta, 1/n,...
                         Czz(:, t), Czx(:, t), deltaMloop, deltaSloop);
                        deltaMz(:, t) = sum(deltaMzloop, 2);
                        deltaSz(:, t) = sum(deltaSzloop, 2);
                        deltaMxs(:, t) = sum(deltaMxsloop, 2);
                        deltaSxs(:, t) = sum(deltaSxsloop, 2);
                    end
                end
            else
                if isempty(Sxs)
                    for t = 1:rB
                        Czz = (J(:, t).*Sz(:, t))/n;
                        if overlap == 0
                            deltaMzloop = reshape(repmat(reshape(repmat(transpose(deltaM(:, t)),...
                                [ki, 1]), [ki * ho, wo * fo * B]), [ki, 1]),...
                                [ho * wo * fo * ki * ki * B, 1]);
                            deltaSzloop = reshape(repmat(reshape(repmat(transpose(deltaS(:, t)),...
                                [ki, 1]), [ki * ho, wo * fo * B]), [ki, 1]),...
                                [ho * wo * fo * ki * ki * B, 1]);
                        else
                            zeroPadding = zeros(1,1,'like',deltaM);
                            deltaMzloop = [deltaM(:, t); zeroPadding];
                            deltaSzloop = [deltaS(:, t); zeroPadding];
                            deltaMzloop = deltaMzloop(idx);
                            deltaSzloop = deltaSzloop(idx);
                        end
                        deltaMzloop   = Czz .* deltaMzloop;
                        deltaSzloop   = Czz .* deltaSzloop .* Czz;
                        deltaMz(:, t) = sum(deltaMzloop, 2);
                        deltaSz(:, t) = sum(deltaSzloop, 2);
                    end
                else
                    Czz = (J.*Sz)/n;
                    Czx = (J.*Sxs)/n;
                    for t = 1:rB                       
                        if overlap == 0
                            deltaMloop = reshape(repmat(reshape(repmat(transpose(deltaM(:, t)),...
                                [ki, 1]), [ki * ho, wo * fo * B]), [ki, 1]),...
                                [ho * wo * fo * ki * ki * B, 1]);
                            deltaSloop = reshape(repmat(reshape(repmat(transpose(deltaS(:, t)),...
                                [ki, 1]), [ki * ho, wo * fo * B]), [ki, 1]),...
                                [ho * wo * fo * ki * ki * B, 1]);
                        else
                            zeroPadding = zeros(1,1,'like',deltaM);
                            deltaMloop = [deltaM(:, t); zeroPadding];
                            deltaSloop = [deltaS(:, t); zeroPadding];
                            deltaMloop = deltaMloop(idx);
                            deltaSloop = deltaSloop(idx);
                        end
                        deltaMzloop    = Czz(:, t) .* deltaMloop;
                        deltaSzloop    = Czz(:, t) .* deltaSloop .* Czz(:, t);
                        deltaMxsloop   = Czx(:, t) .* deltaMloop;
                        deltaSxsloop   = Czx(:, t) .* deltaSloop .* Czx(:, t);
                        deltaMz(:, t)  = sum(deltaMzloop, 2);
                        deltaSz(:, t)  = sum(deltaSzloop, 2);
                        deltaMxs(:, t) = sum(deltaMxsloop, 2);
                        deltaSxs(:, t) = sum(deltaSxsloop, 2);
                    end
                end
            end
        end        
        function [mz, Sz, maxIdx] = mpMeanVarMexcuda(maS, ma, Sa,...
                idxPooling, wo, ho, fo, wi, hi, fi, ki, B)
            [mz, Sz, maxIdx] = mpHiddenStateForwardPass4matlab(maS, ma, Sa,...
                idxPooling, wo, ho, fo, wi, hi, fi, ki, B, overlap); 
        end
        function [mz, Sz] = apMeanVarMexcuda(ma, Sa, idxPooling, wo, ho, fo,...
                wi, hi, fi, ki, B, overlap)
            [mz, Sz] = apHiddenStateForwardPass4matlab(ma, Sa, idxPooling,...
                wo, ho, fo, wi, hi, fi, ki, B, overlap);            
        end 
        function [deltaMz, deltaSz,...
                deltaMxs, deltaSxs] = apHiddenStateBackwardPassMexcuda(Sz,...
                Sxs, J, deltaM, deltaS, idxSzzUd, wo, ho, fo, wi, hi, fi,...
                ki, B, overlap) 
            [deltaMz, deltaSz] = apHiddenStateBackwardPass4matlab(Sz, J,...
             deltaM, deltaS, idxSzzUd, wo, ho, fo, wi, hi, fi, ki, B, overlap);  
            if ~isempty(Sxs)
                [deltaMxs, deltaSxs] = apHiddenStateBackwardPass4matlab(Sxs,...
                 J, deltaM, deltaS, idxSzzUd, wo, ho, fo, wi, hi, fi, ki, B,...
                 overlap);                
            else
                deltaMxs = [];
                deltaSxs = [];
            end
        end
                
        % Normalization layer
        function [m, S] = pMeanVar(pm, pS, ni, wi, hi, fi, B, rB, li, lo,...
                le)
            if li == le.fc && lo == le.ln 
                pm = reshape(pm, [ni, B * rB]);
                pS = reshape(pS, [ni, B * rB]);
                m  = mean(pm, 1);
                S  = (sum(pS, 1) + sum((pm - m).^2, 1)) / (ni-1);            
            elseif li == le.fc && lo == le.bn
                pm = reshape(pm, [ni, B * rB]);
                pS = reshape(pS, [ni, B * rB]);
                m  = mean(pm, 2);
                S  = (sum(pS, 2) + sum((pm - m).^2, 2)) / (B * rB - 1);
            elseif li ~= le.fc && lo == le.ln
                pm = reshape(pm, [wi * hi * fi, B * rB]);
                pS = reshape(pS, [wi * hi * fi, B * rB]);
                m  = mean(pm, 1);
                S  = (sum(pS, 1) + sum((pm - m).^2, 1))/(wi * hi * fi - 1);
            elseif li ~= le.fc && lo == le.bn
                pm = reshape(reshape(pm, [wi * hi * fi, B * rB])',...
                    [wi * hi * B * rB, fi]);
                pS = reshape(reshape(pS, [wi * hi * fi, B * rB])',...
                    [wi * hi * B * rB, fi]);
                m  = mean(pm, 1);
                S  = (sum(pS, 1) + sum((pm - m).^2, 1)) / (wi * hi * B * rB - 1);
            end
            m = m(:);
            S = S(:);
        end
        function m = distributeNormMeanVar(m, ni, wi, hi, fi, B, rB, li, lo,...
                le)
            if li == le.fc && lo == le.ln                 
                m  = reshape(repmat(m', [ni, 1]), [ni * B, rB]);                         
            elseif li == le.fc && lo == le.bn
                m  = repmat(m, [B, rB]);
            elseif li ~= le.fc && lo == le.ln
                m  = reshape(repmat(m', [wi * hi * fi, 1]),...
                    [wi * hi * fi * B, rB]);
            elseif li ~= le.fc && lo == le.bn
                m  = repmat(reshape(repmat(m', [wi * hi, 1]),...
                    [wi * hi * fi, 1]), [B, rB]);
            end
        end
        function [mz, Sz] = fcNormMeanVar(mz, Sz, mw, Sw, mb, Sb, ma, Sa,...
                mhat, Shat, epsilon, B, rB, gpu)
            mb = repmat(mb, [B, 1]);
            Sb = repmat(Sb, [B, 1]);
            mw = repmat(mw, [B, 1]);            
            Sw = repmat(Sw, [B, 1]);                     
            if gpu
                funA = @(x, y) 1 ./ (x + y);
                A = arrayfun(funA, Shat, epsilon);
                [mz, Sz] = arrayfun(@vectorizedNormMeanVar, ma, Sa, mhat,...
                    mw, Sw, mb, Sb, A);
            else
                for t = 1:rB
                    A = 1 ./ (Shat(:, t) + epsilon);
                    mz(:, t) = sqrt(A) .* (ma(:, t) - mhat(:, t)) .* mw + mb;
                    Sz(:, t) = A .* (Sa(:, t) .* (mw.^2) + Sw .* (ma(:, t).^2 ... 
                        - mhat(:, t).^2 + Sa(:, t))) + Sb;
                end
            end
        end
        function [mz, Sz] = convNormMeanVar(mz, Sz, mw, Sw, mb, Sb, ma, Sa,...
                mhat, Shat, epsilon,  wi, hi, fi, B, rB, gpu)
            mb   = repmat(reshape(repmat(mb', [wi * hi, 1]), [fi * hi * wi, 1]), [B, 1]);
            Sb   = repmat(reshape(repmat(Sb', [wi * hi, 1]), [fi * hi * wi, 1]), [B, 1]);      
            mw   = repmat(reshape(repmat(mw', [wi * hi, 1]), [fi * wi * hi, 1]), [B, 1]);
            Sw   = repmat(reshape(repmat(Sw', [wi * hi, 1]), [fi * wi * hi, 1]), [B, 1]);                    
            if gpu
                funA = @(x, y) 1 ./ (x + y);
                A = arrayfun(funA, Shat, epsilon);
                [mz, Sz] = arrayfun(@vectorizedNormMeanVar, ma, Sa, mhat,...
                    mw, Sw, mb, Sb, A);
            else
                for t = 1:rB
                    A = 1 ./ (Shat + epsilon);
                    mz(:, t) = sqrt(A) .* (ma(:, t) - mhat(:, t)) .* mw + mb;
                    Sz(:, t) = A .* (Sa(:, t) .* (mw.^2) + Sw .* (ma(:, t).^2 ...
                        - mhat(:, t).^2 + Sa(:, t))) + Sb;
                end
            end
        end          
        function [deltaMw, deltaSw,...
                deltaMb, deltaSb] = normParameterBackwardPass(deltaMw,...
                deltaSw, deltaMb, deltaSb, Sw, Sb, ma, mra, Sra, epsilon,...
                deltaM, deltaS, ni, wi, hi, fi, B, rB, li, layerEncoder, gpu)
            fun = @(x,y,z,t,q) sqrt((1 ./ (x + q))) .* (y - z) .* t;
            if li == layerEncoder.fc % Previous layer is full-connected
                Sw = repmat(Sw, [B, 1]);
                Cbz = repmat(Sb, [B, 1]);                 
                if gpu                   
                    Cwz = arrayfun(fun, Sra, ma, mra, Sw, epsilon);
                    % Weights
                    for t = 1 : rB 
                        [deltaMwloop, deltaSwloop, deltaMbloop,...
                         deltaSbloop] = arrayfun(@vectorizedDelta4normParam,...
                         Cwz(:, t), Cbz, deltaM(:, t), deltaS(:, t));
                        deltaMw(:, t) = sum(reshape(deltaMwloop, [ni, B]), 2);
                        deltaSw(:, t) = sum(reshape(deltaSwloop, [ni, B]), 2);
                        deltaMb(:, t) = sum(reshape(deltaMbloop, [ni, B]), 2);
                        deltaSb(:, t) = sum(reshape(deltaSbloop, [ni, B]), 2);
                    end
                else
                    for t = 1 : rB
                        A   = 1 ./ (Sra(:, t) + epsilon);
                        Cwz = sqrt(A) .* (ma(:, t) - mra(:, t)) .* Sw;
                        deltaMwloop = Cwz .* deltaM(:, t);
                        deltaSwloop = Cwz .* deltaS(:, t) .* Cwz;
                        deltaMbloop = Cbz .* deltaM(:, t);
                        deltaSbloop = Cbz .* deltaS(:, t) .* Cbz;
                        deltaMw(:, t) = sum(reshape(deltaMwloop, [ni, B]), 2);
                        deltaSw(:, t) = sum(reshape(deltaSwloop, [ni, B]), 2);
                        deltaMb(:, t) = sum(reshape(deltaMbloop, [ni, B]), 2);
                        deltaSb(:, t) = sum(reshape(deltaSbloop, [ni, B]), 2);
                    end
                end
            elseif li == layerEncoder.conv || li == layerEncoder.tconv % Previous layer is convolutional
                Sw  = repmat(reshape(repmat(Sw', [wi * hi, 1]),...
                    [fi * hi * wi, 1]), [B, 1]);
                Cbz = repmat(reshape(repmat(Sb', [wi * hi, 1]),...
                    [fi * hi * wi, 1]), [B, 1]);
                if gpu
                    Cwz = arrayfun(fun, Sra, ma, mra, Sw, epsilon);
                    for t = 1:rB                       
                        [deltaMwloop, deltaSwloop, deltaMbloop,...
                         deltaSbloop] = arrayfun(@vectorizedDelta4normParam,...
                         Cwz(:, t), Cbz, deltaM(:, t), deltaS(:, t));
                        deltaMwloop = squeeze(permute(reshape(deltaMwloop,...
                            [wi * hi, 1, fi, B]),[1 2 4 3]));
                        deltaSwloop = squeeze(permute(reshape(deltaSwloop,...
                            [wi * hi, 1, fi, B]), [1 2 4 3]));
                        deltaMwloop = sum(sum(deltaMwloop, 1), 2);
                        deltaSwloop = sum(sum(deltaSwloop, 1), 2);
                        deltaMw(:, t) = deltaMwloop(:);
                        deltaSw(:, t) = deltaSwloop(:);
                        % Bias
                        deltaMbloop = squeeze(permute(reshape(deltaMbloop,...
                            [wi * hi, 1, fi, B]), [1 2 4 3]));
                        deltaSbloop = squeeze(permute(reshape(deltaSbloop,...
                            [wi * hi, 1, fi, B]), [1 2 4 3]));
                        deltaMbloop = sum(sum(deltaMbloop, 1), 2);
                        deltaSbloop = sum(sum(deltaSbloop, 1), 2);
                        deltaMb(:, t) = deltaMbloop(:);
                        deltaSb(:, t) = deltaSbloop(:);
                    end
                else
                    for t = 1:rB
                        A   = 1 ./ (Sra(:, t) + epsilon);
                        Cwz = sqrt(A) .* (ma(:, t) - mra(:, t)) .* Sw;
                        [deltaMwloop, deltaSwloop] = vectorizedDelta(Cwz,...
                            deltaM(:, t), deltaS(:, t));
                        [deltaMbloop, deltaSbloop] = vectorizedDelta(Cbz,...
                            deltaM(:, t), deltaS(:, t));
                        deltaMwloop = squeeze(permute(reshape(deltaMwloop,...
                            [wi * hi, 1, fi, B]), [1 2 4 3]));
                        deltaSwloop = squeeze(permute(reshape(deltaSwloop,...
                            [wi * hi, 1, fi, B]), [1 2 4 3]));
                        deltaMwloop = sum(sum(deltaMwloop, 1), 2);
                        deltaSwloop = sum(sum(deltaSwloop, 1), 2);
                        deltaMw(:, t) = deltaMwloop(:);
                        deltaSw(:, t) = deltaSwloop(:);
                        % Bias
                        deltaMbloop = squeeze(permute(reshape(deltaMbloop,...
                            [wi * hi, 1, fi, B]),[1 2 4 3]));
                        deltaSbloop = squeeze(permute(reshape(deltaSbloop,...
                            [wi * hi, 1, fi, B]),[1 2 4 3]));
                        deltaMbloop = sum(sum(deltaMbloop, 1), 2);
                        deltaSbloop = sum(sum(deltaSbloop, 1), 2);
                        deltaMb(:, t) = deltaMbloop(:);
                        deltaSb(:, t) = deltaSbloop(:);
                    end
                end
            end
            deltaMw = sum(deltaMw, 2);
            deltaSw = sum(deltaSw, 2);
            deltaMb = sum(deltaMb, 2);
            deltaSb = sum(deltaSb, 2);
        end
        function [deltaMz, deltaSz,...
                deltaMxs, deltaSxs] = normHiddenStateBackwardPass(Sz, Sxs,...
                J, mw, Sra, epsilon, deltaM, deltaS, wi, hi, fi, B, rB, li,...
                layerEncoder, gpu)
            deltaMz = Sz;
            deltaSz = Sz;
            deltaMxs = Sxs;
            deltaSxs = Sxs;
            if li == layerEncoder.fc
                mw = repmat(mw, [B, 1]);                
            elseif li == layerEncoder.conv || li == layerEncoder.tconv
                mw = repmat(reshape(repmat(mw', [wi * hi, 1]),...
                    [fi * wi * hi, 1]), [B, 1]);
            end
            if gpu
                if isempty(Sxs)
                    fun = @(x, y, z, t, q) x .* sqrt(1 ./ (y + q)) .* z .* t;
                    Czz = arrayfun(fun, J, Sra, Sz, mw, epsilon);
                    for t = 1:rB
                        [deltaMz(:, t),...
                         deltaSz(:, t)] = arrayfun(@vectorizedDelta,...
                         Czz(:, t), deltaM(:, t), deltaS(:, t));
                    end
                else
                    fun = @(x, y, z, q) x .* sqrt(1 ./ (y + q)) .* z;
                    Czz = arrayfun(fun, J, Sra, Sz, epsilon);
                    Czx = arrayfun(fun, J, Sra, Sxs, epsilon);
                    for t = 1 : rB
                        [deltaMz(:, t), deltaSz(:, t), deltaMxs(:, t),...
                         deltaSxs(:, t)] = arrayfun(@vectorized4delta, mw,...
                         Czz(:, t), Czx(:, t), deltaM(:, t), deltaS(:, t));
                    end
                end
            else
                if isempty(Sxs)
                    A   = 1 ./ (Sra + epsilon);
                    Czz = J .* sqrt(A) .* Sz .* mw;
                    for t = 1:rB                      
                        [deltaMz(:, t),...
                         deltaSz(:, t)] = vectorizedDelta(Czz, deltaM(:, t),...
                         deltaS(:, t));
                    end
                else
                    A   = 1 ./ (Sra + epsilon);
                    Czz = J .* sqrt(A) .* Sz;
                    Czx = J .* sqrt(A) .* Sz;
                    for t = 1 : rB
                        [deltaMz(:, t), deltaSz(:, t), deltaMxs(:, t),...
                         deltaSxs(:, t)] = arrayfun(@vectorized4delta, mw,...
                         Czz(:, t), Czx(:, t), deltaM(:, t), deltaS(:, t));
                    end
                end
            end
        end
        
        function [m, S] = pMeanVarMexcuda(ma, Sa, ni, wi, hi, fi, B, li,...
                lo, le)
            wi = double(wi);
            hi = double(hi);
            fi = double(fi);   
            B  = double(B);
            li = double(li);
            lo = double(lo);
            if li == le.fc
                if lo == le.ln; N = B; else; N = ni; end
                [m, S] = fcNormMeanVar4matlab(ma, Sa, ni, B, N, lo);
            else
                if lo == le.ln; N = B; else; N = fi; end
                [m, S] = convNormMeanVar4matlab(ma, Sa, wi, hi, fi, B, N, lo);
            end
        end
        function [mra, Sra] = pMeanVarMexcuda_V2(ma, Sa, mraprev, Sraprev,...
                ni, wi, hi, fi, B, li, lo, le, momentum)
            [mra, Sra] = normMeanVar4matlab(ma, Sa, mraprev, Sraprev, ni,...
                wi, hi, fi, B, li, lo, le.fc, le.conv, momentum);
        end
        function [mz, Sz] = fcNormMeanVarMexcuda(mw, Sw, mb, Sb, ma, Sa,...
                mhat, Shat, epsilon, wstartpos, bstartpos, ni, B, lo)
            ni = double(ni);  
            B  = double(B);
            lo = double(lo);
            wstartpos = double(wstartpos);
            bstartpos = double(bstartpos);
            [mz, Sz] = fcNormHiddenStateForwardPass4matlab(mw, Sw, mb, Sb,...
                ma, Sa, mhat, Shat, epsilon, wstartpos, bstartpos, ni, B, lo);
        end
        function [mz, Sz] = convNormMeanVarMexcuda(mw, Sw, mb, Sb, ma, Sa,...
                mhat, Shat, epsilon, wstartpos, bstartpos, wi, hi, fi, B, lo)
            [mz, Sz] = cnnNormHiddenStateForwardPass4matlab(mw, Sw, mb, Sb,...
                ma, Sa, mhat, Shat, epsilon, wstartpos, bstartpos, wi, hi, fi, B, lo);
        end  
        function [deltaMw, deltaSw, deltaMb,...
                deltaSb] = normParameterBackwardPassMexcuda(deltaMw,...
                deltaSw, deltaMb, deltaSb, Sw, Sb, ma, mhat, Shat, deltaM,...
                deltaS, epsilon, wstartpos, bstartpos, ni, wi, hi, fi, B,...
                li, lo, le)
            if li == le.fc % Previous layer is full-connected
                [deltaMw, deltaSw, deltaMb,...
                    deltaSb] = fcNormParamBackwardPass4matlab(Sw, Sb, ma,...
                    mhat, Shat, deltaM, delta, epsilon, wstartpos, bstartpos,...
                    ni, B, lo, deltaMw, deltaSw, deltaMb, deltaSb);           
            elseif li == le.conv || li == le.tconv % Previous layer is convolutional                
                [deltaMw, deltaSw, deltaMb,...
                    deltaSb] = convNormParamBackwardPass4matlab(Sw, Sb, ma,...
                    mhat, Shat, deltaM, deltaS, epsilon, wstartpos, bstartpos,...
                    wi, hi, fi, B, lo, deltaMw, deltaSw, deltaMb, deltaSb);
            end
        end
        function [deltaMz, deltaSz,...
                deltaMxs, deltaSxs] = normHiddenStateBackwardPassMexcuda(Sz,...
                Sxs, J, mw, Shat, deltaM, deltaS, epsilon, wstartpos, ni,...
                wi, hi, fi, B, li, lo, le)
            if li == le.fc      
                [deltaMz,...
                 deltaSz] = fcNormHiddenStateBackwardPass4matlab(mw, Sz, J,...
                 Shat, deltaM, deltaS, epsilon, wstartpos, ni, B, lo);
            elseif li == le.conv || li == le.tconv
                [deltaMz,...
                 deltaSz] = convNormHiddenStateBackwardPass4matlab(mw, Sz,...
                 J, Shat, deltaM, deltaS, epsilon, wstartpos, wi, hi, fi, B,...
                 lo);
            end
            if ~isempty(Sxs)
                if li == le.fc
                    [deltaMxs,...
                     deltaSxs] = fcNormHiddenStateBackwardPass4matlab(mw,...
                     Sxs, J, Shat, deltaM, deltaS, epsilon, wstartpos, ni,...
                     B, lo);
                elseif li == le.conv || li == le.tconv
                    [deltaMxs,...
                     deltaSxs] = convNormHiddenStateBackwardPass4matlab(mw,...
                     Sxs, J, Shat, deltaM, deltaS, epsilon, wstartpos, wi,...
                     hi, fi, B, lo);
                end
            else
                deltaMxs = [];
                deltaSxs = [];
            end
        end
        
        % Full connected layer 
        function [mz, Sz] = fcMeanVar(mz, Sz, mw, Sw, mb, Sb, ma, Sa, ni,...
                no, B, rB, gpu)
            idxSum = 1;
            if any(isnan(mb))
                mb = zeros(1,1,'like', mw);
                Sb = zeros(1,1,'like', Sw);               
            else
                mb = repmat(mb, [B, 1]);
                Sb = repmat(Sb, [B, 1]);
            end
            mw  = repmat(reshape(mw, [ni, no]), [1, B]);                     
            Sw  = repmat(reshape(Sw, [ni, no]), [1, B]);                                  
            if gpu
                for t = 1:rB
                    maloop = reshape(repmat(reshape(ma(:, t), [ni, B]),...
                        [no, 1]), [ni, no*B]);
                    Saloop = reshape(repmat(reshape(Sa(:, t), [ni, B]),...
                        [no, 1]), [ni, no*B]);
                    [mzloop, Szloop] = arrayfun(@vectorizedMeanVar, maloop,...
                        mw, Saloop, Sw);
                    mzloop = transpose(sum(mzloop, idxSum));
                    Szloop = transpose(sum(Szloop, idxSum));
                    [mz(:, t), Sz(:, t)] = arrayfun(@twoPlus, mzloop,...
                        Szloop, mb, Sb);
                end
            else
                for t = 1:rB
                    maloop = reshape(repmat(reshape(ma(:, t), [ni, B]),...
                        [no, 1]), [ni, no*B]);
                    Saloop = reshape(repmat(reshape(Sa(:, t), [ni, B]),...
                        [no, 1]), [ni, no*B]);
                    [mzloop, Szloop] = vectorizedMeanVar(maloop, mw, Saloop,...
                        Sw);
                    mzloop = transpose(sum(mzloop, idxSum));
                    Szloop = transpose(sum(Szloop, idxSum));
                    mz(:, t) = mzloop + mb;
                    Sz(:, t) = Szloop + Sb;
                end
            end            
        end
        function [mz, Sz] = fcMeanVarB1(mw, Sw, mb, Sb, ma, Sa, ni, no,...
                gpu)
            if any(isnan(mb))
                mb = zeros(1,1,'like', mw);
                Sb = zeros(1,1,'like', Sw);               
            end
            mw = reshape(mw, [ni, no]);                     
            Sw = reshape(Sw, [ni, no]); 
            if gpu
                [mzloop, Szloop] = arrayfun(@vectorizedMeanVar, ma, mw,...
                    Sa, Sw);
                mzloop = sum(mzloop, 1);
                Szloop = sum(Szloop, 1);
                mzloop = mzloop(:);
                Szloop = Szloop(:);
                [mz, Sz] = arrayfun(@twoPlus, mzloop, Szloop, mb, Sb);
            else
                [mzloop, Szloop] = vectorizedMeanVar(ma, mw, Sa, Sw);
                mzloop = transpose(sum(mzloop, 1));
                Szloop = transpose(sum(Szloop, 1));
                mz = mzloop + mb;
                Sz = Szloop + Sb;
            end            
        end  
        function [mz, Sz, Szf] = fcMeanCov(mw, Sw, mb, Sb, ma, Sa, Saf, ni,...
                no, B, gpu)
            idxSum = 1;
            if any(isnan(mb))
                mb = zeros(1,1,'like', mw);
                Sb = zeros(1,1,'like', Sw);               
            else
                mb = repmat(mb, [B, 1]);
                Sb = repmat(Sb, [B, 1]);
            end
            mwd = repmat(reshape(mw, [ni, no]), [1, B]);                     
            Swd = repmat(reshape(Sw, [ni, no]), [1, B]);  
            if gpu
                mad = reshape(repmat(reshape(ma, [ni, B]), [no, 1]),...
                    [ni, no * B]);
                Sad = reshape(repmat(reshape(Sa, [ni, B]), [no, 1]),...
                    [ni, no * B]);
                Sz  = Swd .* mad .* mad + Sad .* Swd;
                mz  = mad .* mwd;
                mz  = transpose(sum(mz, idxSum)) + mb;
                Sz  = transpose(sum(Sz, idxSum)) + Sb;
                
                mw1  = repmat(reshape(mw, [ni, 1, no]), [1, 1, no * B]);
                mw2  = repmat(reshape(repmat(reshape(mw, [ni, no]),...
                    [no, 1]), [1, ni, no * no]), [1 1 B]);
                mw3  = pagefun(@mtimes, mw1, mw2);
                Saf  = reshape(repmat(reshape(Saf, [ni, ni, B]),...
                    [1, no * no, 1]), [ni, ni, no * no * B]);
                Szf  = sum(sum(bsxfun(@times, mw3, Saf), 2), 1);
                Szf  = reshape(Szf, no * no, B);
                Szfd = Szf(1 : no + 1 : no * no, :);
                Sz   = Sz + Szfd(:);
                Szf(1 : no + 1 : no * no, :) = reshape(Sz, [no, B]); 
                Szf = Szf(:);
            else
                mad = reshape(repmat(reshape(ma, [ni, B]), [no, 1]),...
                    [ni, no * B]);
                Sad = reshape(repmat(reshape(Sa, [ni, B]), [no, 1]),...
                    [ni, no * B]);
                Sz  = Swd .* mad .* mad + Sad .* Swd;
                mz  = mad .* mwd;
                mz = transpose(sum(mz, idxSum)) + mb;
                Sz = transpose(sum(Sz, idxSum)) + Sb;
                
                mw1  = repmat(mw, [1, 1, no]);
                mw2  = repmat(reshape(mw, [1, ni, no]), [ni * no, 1, 1]);
                mw3  = mw2 .* mw1;
                mw3  = repmat(mw3, [1, 1, B]);
                Saf  = reshape(repmat(reshape(Saf, [ni, ni, B]),...
                    [no, no, 1]), [ni * no, ni, no * B]);
                Szf  = sum(reshape(sum(mw3 .* Saf, 2), [ni, 1, no * no * B]), 1);
                Szf  = reshape(Szf, no * no, B);
                Szfd = Szf(1 : no + 1 : no * no, :);
                Sz   = Sz + Szfd(:);
                Szf(1 : no + 1 : no * no, :) = reshape(Sz, [no, B]); 
                Szf = Szf(:);
            end
        end
        
        function [deltaMw, deltaSw,...
                deltaMb, deltaSb] = fcParameterBackwardPass(deltaMw,...
                deltaSw, deltaMb, deltaSb, Sw, Sb, ma, deltaMr, deltaSr,...
                ni, no, B, rB, gpu)  
            Cbz = repmat(Sb, [1, B]);
            if gpu 
                for t = 1:rB                   
                    maloop   = repmat(reshape(ma(:, t), [ni, B]), [no, 1]);               
                    deltaMrw = reshape(repmat(transpose(deltaMr(:, t)),...
                        [ni, 1]),[ni * no, B]);
                    deltaSrw = reshape(repmat(transpose(deltaSr(:, t)),...
                        [ni, 1]),[ni * no, B]);                  
                    % Weights
                    [deltaMrw, deltaSrw] = arrayfun(@vectorizedDelta_V2, Sw,...
                        maloop, deltaMrw, deltaSrw);
                    deltaMw(:, t) = sum(deltaMrw, 2);
                    deltaSw(:, t) = sum(deltaSrw, 2);
                    % Bias
                    if any(~isnan(Sb))                        
                        deltaMrb = reshape(deltaMr(:, t), [no, B]);
                        deltaSrb = reshape(deltaSr(:, t), [no, B]);                      
                        [deltaMrb, deltaSrb] = arrayfun(@vectorizedDelta,...
                            Cbz, deltaMrb, deltaSrb);
                        deltaMb(:, t) = sum(deltaMrb, 2);
                        deltaSb(:, t) = sum(deltaSrb, 2);
                    end
                end
            else
                for t = 1:rB
                    maloop   = repmat(reshape(ma(:, t), [ni, B]), [no, 1]);               
                    deltaMrw = reshape(repmat(transpose(deltaMr(:, t)),...
                        [ni, 1]),[ni*no, B]);
                    deltaSrw = reshape(repmat(transpose(deltaSr(:, t)),...
                        [ni, 1]),[ni*no, B]); 
                    Cwz      = Sw .* maloop;
                    deltaMrw = Cwz .* deltaMrw;
                    deltaSrw = Cwz .* deltaSrw .* Cwz;
                    deltaMw(:, t) = sum(deltaMrw, 2);
                    deltaSw(:, t) = sum(deltaSrw, 2);
                    if any(~isnan(Sb))
                        deltaMrb = reshape(deltaMr(:, t), [no, B]);
                        deltaSrb = reshape(deltaSr(:, t), [no, B]);                        
                        deltaMrb = Cbz .* deltaMrb;
                        deltaSrb = Cbz .* deltaSrb .* Cbz;
                        deltaMb(:, t) = sum(deltaMrb, 2);
                        deltaSb(:, t) = sum(deltaSrb, 2);
                    end
                end
            end  
            deltaMw = sum(deltaMw, 2);
            deltaSw = sum(deltaSw, 2);
            deltaMb = sum(deltaMb, 2);
            deltaSb = sum(deltaSb, 2);
        end
        function [deltaMw, deltaSw,...
                deltaMb, deltaSb] = fcParameterBackwardPassB1(Sw, Sb, ma,...
                deltaMr, deltaSr, ni, no, gpu)  
            Cbz      = Sb;                   
            maloop   = repmat(ma, [no, 1]);
            deltaMrw = repmat(transpose(deltaMr), [ni, 1]);
            deltaMrw = deltaMrw(:);
            deltaSrw = repmat(transpose(deltaSr), [ni, 1]);
            deltaSrw = deltaSrw(:);
            % Weights
            if gpu
                [deltaMrw, deltaSrw] = arrayfun(@vectorizedDelta_V2, Sw,...
                    maloop, deltaMrw, deltaSrw);
            else
                Cwa = Sw .* maloop;
                deltaMrw = Cwa .* deltaMrw;
                deltaSrw = (Cwa.^2) .* deltaSrw;                
            end
            deltaMw = sum(deltaMrw, 2);
            deltaSw = sum(deltaSrw, 2);
            % Bias
            if any(~isnan(Sb))
                if gpu
                    [deltaMrb, deltaSrb] = arrayfun(@vectorizedDelta, Cbz,...
                        deltaMr, deltaSr);
                else
                    [deltaMrb, deltaSrb] = vectorizedDelta(Cbz, deltaMr, deltaSr);
                end
                deltaMb = sum(deltaMrb, 2);
                deltaSb = sum(deltaSrb, 2);
            else
                deltaMb = Sb;
                deltaSb = Sb;
            end
        end
        function [deltaMz, deltaSz,...
                deltaMzx, deltaSzx] = fcHiddenStateBackwardPass(Sz, Sxs,...
                J, mw, deltaM, deltaS, ni, no, B, rB, gpu) 
            deltaMz  = Sz;
            deltaSz  = Sz;
            deltaMzx = Sxs;
            deltaSzx = Sxs;
            mw = repmat(reshape(mw, [ni, no]), [B, 1]);              
            if gpu
                Caz = bsxfun(@times, J, Sz);
                if isempty(Sxs)
                    for t = 1 : rB
                        deltaMzloop = reshape(repmat(reshape(deltaM(:, t),...
                            [no, B]), [ni, 1]), [no, ni * B])';
                        deltaSzloop = reshape(repmat(reshape(deltaS(:, t),...
                            [no, B]), [ni, 1]), [no, ni * B])';
                        [deltaMzloop,...
                         deltaSzloop] = arrayfun(@vectorizedDelta_V2, mw,...
                         Caz(:, t), deltaMzloop, deltaSzloop);
                        deltaMz(:, t) = sum(deltaMzloop, 2);
                        deltaSz(:, t) = sum(deltaSzloop, 2);
                    end
                else
                    Caxs = bsxfun(@times, J, Sxs);
                    for t = 1 : rB
                        deltaMloop = reshape(repmat(reshape(deltaM(:, t),...
                            [no, B]), [ni, 1]), [no, ni * B])';
                        deltaSloop = reshape(repmat(reshape(deltaS(:, t),...
                            [no, B]), [ni, 1]), [no, ni * B])';
                        [deltaMzloop, deltaSzloop, deltaMxsloop,...
                         deltaSxsloop] = arrayfun(@vectorized4delta, mw,...
                         Caz(:, t), Caxs(:, t), deltaMloop, deltaSloop);
                        deltaMz(:, t)  = sum(deltaMzloop, 2);
                        deltaSz(:, t)  = sum(deltaSzloop, 2);
                        deltaMzx(:, t) = sum(deltaMxsloop, 2);
                        deltaSzx(:, t) = sum(deltaSxsloop, 2);
                    end
                end
            else
                if isempty(Sxs)
                    for t = 1 : rB
                        Czz = J(:, t) .* Sz(:, t) .* mw;
                        deltaMzloop = reshape(repmat(reshape(deltaM(:, t),...
                            [no, B]), [ni, 1]), [no, ni*B])';
                        deltaSzloop = reshape(repmat(reshape(deltaS(:, t),...
                            [no, B]), [ni, 1]), [no, ni*B])';
                        deltaMzloop = Czz .* deltaMzloop;
                        deltaSzloop = Czz .* deltaSzloop .* Czz;
                        deltaMz(:, t) = sum(deltaMzloop, 2);
                        deltaSz(:, t) = sum(deltaSzloop, 2);
                    end
                else
                    for t = 1:rB
                        Czz = J(:, t) .*Sz (:, t) .* mw;
                        Czx = J(:, t) .*Sz (:, t) .* mw;
                        deltaMloop     = reshape(repmat(reshape(deltaM(:, t),...
                            [no, B]), [ni, 1]), [no, ni*B])';
                        deltaSloop     = reshape(repmat(reshape(deltaS(:, t),...
                            [no, B]), [ni, 1]), [no, ni*B])';
                        deltaMzloop    = Czz .* deltaMloop;
                        deltaSzloop    = Czz .* deltaSloop .* Czz;
                        deltaMxsloop   = Czx .* deltaMloop;
                        deltaSxsloop   = Czx .* deltaSloop .* Czx;
                        deltaMz(:, t)  = sum(deltaMzloop, 2);
                        deltaSz(:, t)  = sum(deltaSzloop, 2);
                        deltaMzx(:, t) = sum(deltaMxsloop, 2);
                        deltaSzx(:, t) = sum(deltaSxsloop, 2);
                    end
                end
            end
        end 
        function [deltaMz, deltaSz,...
                deltaMzx, deltaSzx] = fcHiddenStateBackwardPassB1(Sz, Sxs,...
                J, mw, deltaM, deltaS, ni, no, gpu) 
            mw  = reshape(mw, [ni, no]);              
            deltaMzx = Sxs;
            deltaSzx = Sxs;
            if isempty(Sxs)
                deltaMloop = repmat(deltaM', [ni, 1]);
                deltaSloop = repmat(deltaS', [ni, 1]);
                if gpu
                    Caz = bsxfun(@times, J, Sz);
                    [deltaMzloop,...
                     deltaSzloop] = arrayfun(@vectorizedDelta_V2, mw, Caz,...
                     deltaMloop, deltaSloop);
                else
                    Caz = J .* Sz;
                    Cwa = mw .* Caz;
                    deltaMzloop = Cwa .* deltaMloop;
                    deltaSzloop = (Cwa.^2) .* deltaSloop;
                end
                deltaMz = sum(deltaMzloop, 2);
                deltaSz = sum(deltaSzloop, 2);
            else                
                deltaMloop = repmat(deltaM', [ni, 1]);
                deltaSloop = repmat(deltaS', [ni, 1]);
                if gpu
                    Caz = bsxfun(@times, J, Sz);
                    Caxs = bsxfun(@times, J, Sxs);
                    [deltaMzloop, deltaSzloop, deltaMxsloop,...
                     deltaSxsloop] = arrayfun(@vectorized4delta, mw, Caz,...
                     Caxs, deltaMloop, deltaSloop);
                else
                    Caz = J .* Sz;
                    Caxs = J .* Sxs;
                    [deltaMzloop, deltaSzloop, deltaMxsloop,...
                     deltaSxsloop] = vectorized4delta(mw, Caz, Caxs,...
                     deltaMloop, deltaSloop);
                end
                deltaMz  = sum(deltaMzloop, 2);
                deltaSz  = sum(deltaSzloop, 2);
                deltaMzx = sum(deltaMxsloop, 2);
                deltaSzx = sum(deltaSxsloop, 2);
            end
        end   
        
        function [mz, Sz] = fcMeanVarMexcuda(mw, Sw, mb, Sb, ma, Sa, widx,...
                bidx, ni, no, B)
            [mz, Sz] = fcMeanVar4matlab(mw, Sw, mb, Sb, ma, Sa, widx, bidx,...
                no, ni, B);
        end
        function [deltaMz, deltaSz,...
                deltaMzx, deltaSzx] = fcHiddenStateBackwardPassMexcuda(Sz,...
                J, mw, deltaM, deltaS, widx, ni, no, B) 
            [deltaMz, deltaSz] = fcHiddenStateBackwardPass4matlab(mw, Sz, J,...
                deltaM, deltaS, widx, ni, no, B);
            deltaMzx = [];
            deltaSzx = [];
        end 
        function [deltaMw, deltaSw,...
                deltaMb, deltaSb] = fcParameterBackwardPassMexcuda(Sw, Sb,...
                ma, deltaM, deltaS, widx, bidx, ni, no, B, deltaMw, deltaSw,...
                deltaMb, deltaSb)  
            [deltaMw, deltaSw, deltaMb,...
             deltaSb] = fcParamBackwardPass4matlab(Sw, ma, Sb, deltaM,...
             deltaS, widx, bidx, ni, B, no, no, B, 1, deltaMw, deltaSw,...
             deltaMb, deltaSb);
        end
        
        % FC Derivative
        function [mdgi, Sdgi, Cdx] = fcDerivative(mw, Sw, mwo, Jo, J, mao,...
                Sao, mai, Sai, Szi, mdai, Sdai, mdgo, mdgoe, Sdgo, mdgo2,...
                acto, acti, ni, no, no2, B)
            [mpdi, Spdi]  = tagi.fcMeanVarDnode(mw, Sw, mdai, Sdai, ni, no,...
                B);
            [Caow, Caoai] = tagi.fcCovawaa(mw, Sw, Jo, mai, Sai, ni, no, B);
            [Cdow, Cdodi] = tagi.fcCovdwddd(mao, mai, Caow, Caoai, acto,...
                acti, ni, no, B);
            Cdowdi        = tagi.fcCovdwd(mdai, mw, Cdow, Cdodi, ni, no, B);
            Cdgodgi       = tagi.fcCovDlayer(mdgo2, mwo, Cdowdi, ni, no,...
                no2, B);
            [mdgi, Sdgi]  = tagi.fcMeanVarDlayer(mpdi, Spdi, mdgo, mdgoe,...
                Sdgo, Cdgodgi, ni, no, no2, B);
                        
            [Caizi, Caozi] = tagi.fcCovaz(Jo, J, Szi, mw, ni, no, B);
            [Cdozi, Cdizi] = tagi.fcCovdz(mao, mai, Caizi, Caozi, acto,...
                acti, ni, no, B);
            Cdx            = tagi.covdx(mwo, mw, mdgo2, mpdi, mdgoe, Cdozi,...
                Cdizi, ni, no, no2, B);
        end
        function [md, Sd] = fcMeanVarDnode(mw, Sw, mda, Sda, ni, no, B)
            mw = repmat(reshape(mw, [ni, no]), [B, 1]);                     
            Sw = repmat(reshape(Sw, [ni, no]), [B, 1]); 
            md = mw .* mda;
            Sd = Sw .* Sda + Sw .* (mda .^2) + Sda .* (mw .^2);
        end       
        function [Caw, Caa] = fcCovawaa(mw, Sw, Jo, mai, Sai, ni, no, B)
            Joloop = reshape(repmat(reshape(Jo, [no, B]), [ni, 1]),...
                [no, ni*B])'; 
            Sw  = repmat(reshape(Sw, [ni, no]), [B, 1]); 
            Caw = Sw .* mai .* Joloop;
            
            mw = repmat(reshape(mw, [ni, no]), [B, 1]);               
            Caa = mw .* Sai .* Joloop;
        end
        function [Cdow, Cdodi] = fcCovdwddd(mao, mai, Caow, Caoai, acto,...
                acti, ni, no, B)
            mao = reshape(repmat(reshape(mao, [no, B]), [ni, 1]), [no, ni*B])';
            
            if acti==1 % tanh
                Cdodi = 2 * Caoai .^2 + 4 * Caoai .* mai .* mao;
            elseif acti==2 % sigmoid
                Cdodi = Caoai - 2 * Caoai .* mai - 2 * mao .* Caoai...
                    + 2 * Caoai .^2 + 4 * Caoai .* mai .* mao;
            elseif acti==4 % relu
                Cdodi = zeros(size(mao), 'like', mao);
            elseif acti==0
                Cdodi = zeros(size(mao), 'like', mao);
            end
            
            if acto==1 % tanh
                Cdow = -2 * mao .* Caow;
            elseif acto==2 % sigmoid
                Cdow = Caow .* (1 - 2 * mao);
            elseif acto==4 % relu
                Cdow = zeros(size(mao), 'like', mao);
            elseif acto==0
                Cdow = zeros(size(mao), 'like', mao);
            end
        end                       
        function Cdowdi = fcCovdwd(md, mw, Cdow, Cdodi, ni, no, B)
            mw = repmat(reshape(mw, [ni, no]), [B, 1]);
            Cdowdi = Cdow .* md + Cdodi .* mw;
        end
        function Cdgodgi = fcCovDlayer(mdgo2, mwo, Cdowdi, ni, no, no2, B)
            mdgo2   = reshape(repmat(mdgo2', [no, 1]), [no*no2, B]);
            m       = reshape(repmat(mdgo2 .* mwo, [ni, 1]), [no*no2, B*ni])';
            Cdowdi  = repmat(Cdowdi, [1 no2]);
            Cdgodgi = Cdowdi .* m;
        end
        function [md, Sd] = fcMeanVarDlayer(mx, Sx, my, mye, Sy, Cxy, ni,...
                no, no2, B)
            my  = reshape(repmat(reshape(my, [no, B]), [ni, 1]), [no, ni * B])';
            Sy  = reshape(repmat(reshape(Sy, [no, B]), [ni, 1]), [no, ni * B])';
            mye = reshape(repmat(reshape(permute(reshape(mye', [no2, no, B]),...
                [2 1 3]), [no*no2, 1, B]), [1 ni, 1]), [no*no2, B*ni])';
            
            md = mx .* my;
            Sd = Sx .* Sy  + Sy .* (mx .^2);
            
            SxS = repmat(Sx, [1, no2]);
            Sd2 = sum(reshape(SxS .* (mye .^2), [B * ni * no, no2]), 2);
            Sd2 = reshape(Sd2, [B * ni, no]);
            
            Sd = Sd + Sd2;
            
            Cxym = sum(reshape(Cxy, [B * ni * no, no2]), 2);
            Cxym = reshape(Cxym, [B * ni, no]);
            md   = md + Cxym;
            
            CxyS1 = sum(reshape(Cxy .^2, [B * ni * no, no2]),2);
            CxyS1 = reshape(CxyS1, [B * ni, no]);
            
            CxyS2 = sum(reshape(2 * Cxy .* mye, [B * ni * no, no2]), 2);
            CxyS2 = reshape(CxyS2, [B*ni, no]);
            CxyS2 = CxyS2 .* mx;
            Sd    = Sd + CxyS1 + CxyS2;
        end
        function [Caizi, Caozi] = fcCovaz(Jo, J, Sz, mw, ni, no, B)
            Jo    = reshape(repmat(reshape(Jo, [no, B]), [ni, 1]),...
                [no, ni * B])';
            mw    = repmat(reshape(mw, [ni, no]), [B, 1]);   
            Caizi = J .* Sz;
            Caozi = Jo .* Caizi .* mw;
        end
        function [Cdozi, Cdizi] = fcCovdz(mao, mai, Caizi, Caozi, acto,...
                acti, ni, no, B)
            mao = reshape(repmat(reshape(mao, [no, B]), [ni, 1]),...
                [no, ni * B])';
            if acti==1 % tanh
                Cdizi = -2 * mai .* Caizi;
            elseif acti==2 % sigmoid               
                Cdizi = (1 - 2 * mai) .* Caizi;
            elseif acti==4 % relu
                Cdizi = zeros(size(mai), 'like', mai);
            else
                Cdizi = zeros(size(mai), 'like', mai);
            end
            
            if acto==1 % tanh
                Cdozi = -2 * mao .* Caozi;
            elseif acto==2 % sigmoid                
                Cdozi = (1 - 2 * mao) .* Caozi;
            elseif acto==4 % relu
                Cdozi = zeros(size(mao), 'like', mao);
            else
                Cdozi = zeros(size(mao), 'like', mao);
            end
        end
        function Cdx = covdx(mwo, mw, mdgo2, mpdi, mdgoe, Cdozi, Cdizi, ni,...
                no, no2, B)
            mdgo2 = reshape(repmat(mdgo2', [no, 1]), [no * no2, B]);
            mw    = repmat(reshape(mw, [ni, no]), [B, 1]);
            mdgoe = reshape(repmat(reshape(permute(reshape(mdgoe', [no2, no, B]),...
                [2 1 3]), [no * no2, 1, B]), [1 ni, 1]), [no * no2, B * ni])';
            m     = reshape(repmat(mdgo2 .* mwo, [ni, 1]), [no * no2, B * ni])';
            Cdozi = repmat(Cdozi, [1, no2]);
            mpdi  = repmat(mpdi, [1, no2]);
            Cdx1  = Cdozi .* m .* mpdi;
            
            mwdx2 = repmat(mw, [1, no2]);
            Cdx2  = Cdizi .* mwdx2 .* mdgoe;
            
            Cdx   = reshape(sum(reshape(Cdx1+Cdx2, [B*ni*no, no2]), 2),...
                [B*ni, no]);
        end
        
        % Convolutional layer
        function [mz, Sz] = convMeanVar(mz, Sz, mw, Sw, mb, Sb, ma, Sa,...
                idxFmwa, ki, fi, wo, ho, fo, B, rB, padding, gpu)           
            mw = repmat(reshape(mw, [ki * ki * fi, 1, fo]),...
                [1, B * wo * ho, 1]);  
            Sw = repmat(reshape(Sw, [ki * ki * fi, 1, fo]),...
                [1, B * wo * ho, 1]);  
            if padding ~= 0 && any(~isempty(Sa))
                zeroPad = zeros(1, size(ma, 2), 'like', ma);
                ma = [ma; zeroPad];
                Sa = [Sa; zeroPad];
            end  
            if gpu
                for t = 1:rB
                    maloop = ma(:, t);
                    Saloop = Sa(:, t);
                    maloop = maloop(idxFmwa{2});
                    Saloop = Saloop(idxFmwa{2});
                    [mzloop, Szloop] = arrayfun(@vectorizedMeanVar, maloop,...
                        mw, Saloop, Sw);
                    mzloop = permute(reshape(sum(mzloop, 1), [wo * ho, B, fo]),...
                        [1 3 2]);
                    Szloop = permute(reshape(sum(Szloop, 1), [wo * ho, B, fo]),...
                        [1 3 2]);
                    mz(:, t) = mzloop(:);
                    Sz(:, t) = Szloop(:);
                end
                if ~any(isnan(mb))
                    mb = repmat(reshape(repmat(mb', [wo*ho, 1]), [wo*ho*fo, 1]),...
                        [B, 1]);
                    Sb = repmat(reshape(repmat(Sb', [wo*ho, 1]), [wo*ho*fo, 1]),...
                        [B, 1]);
                    [mz, Sz] = arrayfun(@twoPlus, mz, Sz, mb, Sb);
                end                
            else
                for t = 1:rB
                    maloop = ma(:, t);
                    Saloop = Sa(:, t);
                    maloop = repmat(maloop(idxFmwa{2}), [1, 1, fo]);
                    Saloop = repmat(Saloop(idxFmwa{2}), [1, 1, fo]);
                    [mzloop, Szloop] = vectorizedMeanVar(maloop, mw, Saloop,...
                        Sw);
                    mzloop = sum(mzloop, 1);
                    Szloop = sum(Szloop, 1);
                    mzloop = permute(reshape(mzloop, [wo*ho, B, fo]), [1 3 2]);
                    Szloop = permute(reshape(Szloop, [wo*ho, B, fo]), [1 3 2]);
                    mz(:, t) = mzloop(:);
                    Sz(:, t) = Szloop(:);
                end
                if ~any(isnan(mb))
                    mb = repmat(reshape(repmat(mb', [wo*ho, 1]), [wo*ho*fo, 1]),...
                        [B, 1]);
                    Sb = repmat(reshape(repmat(Sb', [wo*ho, 1]), [wo*ho*fo, 1]),...
                        [B, 1]);
                    [mz, Sz] = arrayfun(@twoPlus, mz, Sz, mb, Sb);
                end 
            end
        end   
        function [mz, Sz] = convMeanVarB1(mw, Sw, mb, Sb, ma, Sa, idxFmwa,...
                ki, fi, wo, ho, fo, padding, gpu)           
            if any(isnan(mb))
                mb = zeros(1, 1, 'like', mw);
                Sb = zeros(1, 1, 'like', Sw);               
            else
                mb = reshape(repmat(mb', [wo * ho, 1]), [wo * ho * fo, 1]);
                Sb = reshape(repmat(Sb', [wo * ho, 1]), [wo * ho * fo, 1]);
            end
            mw = reshape(mw, [ki * ki * fi, 1, fo]);  
            Sw = reshape(Sw, [ki * ki * fi, 1, fo]);            
            if padding ~= 0 && any(~isempty(Sa))
                zeroPad = zeros(1, size(ma, 2), 'like', ma);
                ma = [ma; zeroPad];
                Sa = [Sa; zeroPad];
            end 
            if gpu
                ma = ma(idxFmwa{2});
                Sa = Sa(idxFmwa{2});
                [mz, Sz] = arrayfun(@vectorizedMeanVar, ma, mw, Sa, Sw);
                mz = sum(mz, 1);
                Sz = sum(Sz, 1);
                mz = mz(:);
                Sz = Sz(:);
                [mz, Sz] = arrayfun(@twoPlus, mz, Sz, mb, Sb);
            else
                maloop = repmat(ma(idxFmwa{2}), [1, 1, fo]);
                Saloop = repmat(Sa(idxFmwa{2}), [1, 1, fo]);
                [mzloop, Szloop] = vectorizedMeanVar(maloop, mw, Saloop, Sw);
                mzloop = sum(mzloop, 1);
                Szloop = sum(Szloop, 1);
                mz = mzloop(:) + mb;
                Sz = Szloop(:) + Sb;
            end
        end
        function [deltaMw, deltaSw,...
                deltaMb, deltaSb] = convParameterBackwardPass(deltaMw,...
                deltaSw, deltaMb, deltaSb, Sw, Sb, ma, deltaMr, deltaSr,...
                idxFmwa, padding, k, fi, wo, ho, fo, B, rB, gpu)    
            Cbz = repmat(Sb', [1, B]);            
            Sw  = reshape(Sw, [k*k*fi, 1, fo]);
            if padding ~= 0
                zeroPad = zeros(1, size(ma, 2), 'like', ma);
                ma = [ma; zeroPad];
            end
            if gpu 
                for t = 1:rB
                    deltaMrw = repmat(reshape(permute(reshape(deltaMr(:, t),...
                        [1, wo * ho, fo, B]), [1, 2, 4, 3]),...
                        [1, wo * ho * B, fo]), [k * k * fi, 1, 1]);
                    deltaSrw = repmat(reshape(permute(reshape(deltaSr(:, t),...
                        [1, wo * ho, fo, B]), [1, 2, 4, 3]),...
                        [1, wo * ho * B, fo]), [k * k * fi, 1, 1]);
                    % Weights
                    maloop = ma(:, t);
                    maloop = maloop(idxFmwa{2});
                    [deltaMrw, deltaSrw] = arrayfun(@vectorizedDelta_V2, Sw,...
                        maloop, deltaMrw, deltaSrw);
                    deltaMrw = sum(deltaMrw, 2);
                    deltaSrw = sum(deltaSrw, 2);
                    deltaMw(:, t) = deltaMrw(:);
                    deltaSw(:, t) = deltaSrw(:);
                    % Bias
                    if any(~isnan(Sb))%||any(~isempty(Sb)) 
                        deltaMrb = reshape(deltaMr(:, t), [ho * wo, fo * B]);
                        deltaSrb = reshape(deltaSr(:, t), [ho * wo, fo * B]);
                        [deltaMrb, deltaSrb] = arrayfun(@vectorizedDelta,...
                            Cbz, deltaMrb, deltaSrb);
                        deltaMb(:, t) = sum(reshape(sum(deltaMrb, 1),...
                            [fo, B]), 2);
                        deltaSb(:, t) = sum(reshape(sum(deltaSrb, 1),...
                            [fo, B]), 2);
                    else
                        deltaMb(:, t) = nan;
                        deltaSb(:, t) = nan;
                    end
                end
            else 
                for t = 1:rB
                    deltaMrw = repmat(reshape(permute(reshape(deltaMr(:, t),...
                        [1, wo * ho, fo, B]), [1, 2, 4, 3]),...
                        [1, wo * ho * B, fo]), [k * k * fi, 1, 1]);
                    deltaSrw = repmat(reshape(permute(reshape(deltaSr(:, t),...
                        [1, wo * ho, fo, B]), [1, 2, 4, 3]),...
                        [1, wo * ho * B, fo]), [k * k * fi, 1, 1]);
                    % Weights
                    maloop = ma(:, t);
                    maloop = repmat(maloop(idxFmwa{2}), [1, 1, fo]);
                    Cwz    = bsxfun(@times, Sw, maloop); 
                    deltaMrw = Cwz .* deltaMrw;
                    deltaSrw = (Cwz.^2) .* deltaSrw;
                    deltaMrw = sum(deltaMrw, 2);
                    deltaSrw = sum(deltaSrw, 2);
                    deltaMw(:, t) = deltaMrw(:);
                    deltaSw(:, t) = deltaSrw(:);
                    if any(~isnan(Sb))
                        deltaMrb = reshape(deltaMr(:, t), [ho * wo, fo * B]);
                        deltaSrb = reshape(deltaSr(:, t), [ho * wo, fo * B]);
                        deltaMrb = Cbz .* deltaMrb;
                        deltaSrb = (Cbz.^2) .* deltaSrb;
                        deltaMb(:, t) = sum(reshape(sum(deltaMrb, 1),...
                            [fo, B]), 2);
                        deltaSb(:, t) = sum(reshape(sum(deltaSrb, 1),...
                            [fo, B]), 2);
                    else
                        deltaMb(:, t) = nan;
                        deltaSb(:, t) = nan;
                    end
                end
            end
            deltaMw = sum(deltaMw, 2);
            deltaSw = sum(deltaSw, 2);
            deltaMb = sum(deltaMb, 2);
            deltaSb = sum(deltaSb, 2);
        end     
        function [deltaMw, deltaSw,...
                deltaMb, deltaSb] = convParameterBackwardPassB1(Sw, Sb, ma,...
                deltaMr, deltaSr, idxFmwa, padding, k, fi, wo, ho, fo, gpu)    
            Cbz = Sb';            
            Sw  = reshape(Sw, [k * k * fi, 1, fo]);
            if padding ~= 0
                zeroPad = zeros(1, size(ma, 2), 'like', ma);
                ma = [ma; zeroPad];
            end
            deltaMrw = reshape(deltaMr, [1, wo*ho, fo]);
            deltaSrw = reshape(deltaSr, [1, wo*ho, fo]);
            % Weights
            ma = ma(idxFmwa{2});
            if gpu
                [deltaMrw, deltaSrw] = arrayfun(@vectorizedDelta_V2, Sw, ma,...
                    deltaMrw, deltaSrw);
            else
                [deltaMrw, deltaSrw] = vectorizedDelta_V2(Sw, ma, deltaMrw,...
                    deltaSrw);
            end
            deltaMrw = sum(deltaMrw, 2);
            deltaSrw = sum(deltaSrw, 2);
            deltaMw  = deltaMrw(:);
            deltaSw  = deltaSrw(:);
            % Bias
            if any(~isnan(Sb))%||any(~isempty(Sb))
                deltaMrb = reshape(deltaMr, [ho * wo, fo]);
                deltaSrb = reshape(deltaSr, [ho * wo, fo]);
                if gpu
                    [deltaMrb, deltaSrb] = arrayfun(@vectorizedDelta, Cbz,...
                        deltaMrb, deltaSrb);
                else
                    [deltaMrb, deltaSrb] = vectorizedDelta(Cbz, deltaMrb,...
                        deltaSrb);
                end
                deltaMb = sum(deltaMrb, 1);
                deltaMb = deltaMb(:);
                deltaSb = sum(deltaSrb, 1);
                deltaSb = deltaSb(:);
            else
                deltaMb = nan;
                deltaSb = nan;
            end
        end  
        function [deltaMz, deltaSz,...
                deltaMzx, deltaSzx] = convHiddenStateBackwardPass(Sz, Sxs,...
                J, mw, deltaM, deltaS, idx, idxFCzwa, wi, hi, fi, B, rB, gpu)
            deltaMz  = Sz;
            deltaSz  = Sz;
            deltaMzx = Sxs;
            deltaSzx = Sxs;
            n = size(idxFCzwa{1}, 1);
            mw = [mw; zeros(1, 1, 'like', mw)];
            mw = repmat(reshape(mw(idxFCzwa{1}), [n, wi * hi, fi]), [1, B, 1]);           
            Caz = bsxfun(@times, Sz, J);
            if ~isempty(idx)
                deltaM = [deltaM; zeros(1, size(deltaM, 2), 'like', deltaM)];
                deltaS = [deltaS; zeros(1, size(deltaS, 2), 'like', deltaS)];
            end
            if gpu  
                if isempty(Sxs)
                    for t = 1:rB
                        deltaMloop = deltaM(:, t);
                        deltaSloop = deltaS(:, t);
                        if ~isempty(idx)
                            deltaMloop = deltaMloop(idx');
                            deltaSloop = deltaSloop(idx');
                        end
                        Cazloop = reshape(permute(reshape(Caz(:, t),...
                            [1, wi * hi, fi, B]), [1, 2, 4, 3]), [1, wi * hi * B, fi]);
                        [deltaMloop,...
                            deltaSloop] = arrayfun(@vectorizedDelta_V2,...
                            Cazloop, mw, deltaMloop, deltaSloop);
                        deltaMloop = permute(reshape(sum(deltaMloop, 1),...
                            [wi * hi, B, fi]), [1, 3, 2]);
                        deltaSloop = permute(reshape(sum(deltaSloop, 1),...
                            [wi * hi, B, fi]), [1, 3, 2]);
                        deltaMz(:, t) = deltaMloop(:);
                        deltaSz(:, t) = deltaSloop(:);
                    end
                else
                    Caxs = bsxfun(@times, Sxs, J);
                    for t = 1:rB
                        deltaMloop = deltaM(:, t);
                        deltaSloop = deltaS(:, t);
                        if ~isempty(idx)
                            deltaMloop = deltaMloop(idx');
                            deltaSloop = deltaSloop(idx');
                        end
                        Cazloop  = reshape(permute(reshape(Caz(:, t),...
                            [1, wi * hi, fi, B]), [1, 2, 4, 3]),...
                            [1, wi * hi * B, fi]);
                        Caxsloop = reshape(permute(reshape(Caxs(:, t),...
                            [1, wi * hi, fi, B]), [1, 2, 4, 3]),...
                            [1, wi * hi * B, fi]);
                        [deltaMzloop, deltaSzloop, deltaMzxloop,...
                         deltaSzxloop] = arrayfun(@vectorized4delta, mw,...
                         Cazloop, Caxsloop, deltaMloop, deltaSloop);
                        deltaMzloop  = permute(reshape(sum(deltaMzloop, 1),...
                            [wi * hi, B, fi]), [1, 3, 2]);
                        deltaSzloop  = permute(reshape(sum(deltaSzloop, 1),...
                            [wi * hi, B, fi]), [1, 3, 2]);
                        deltaMzxloop = permute(reshape(sum(deltaMzxloop, 1),...
                            [wi * hi, B, fi]), [1, 3, 2]);
                        deltaSzxloop = permute(reshape(sum(deltaSzxloop, 1),...
                            [wi * hi, B, fi]), [1, 3, 2]);
                        deltaMz(:, t) = deltaMzloop(:);
                        deltaSz(:, t) = deltaSzloop(:);
                        deltaMzx(:, t) = deltaMzxloop(:);
                        deltaSzx(:, t) = deltaSzxloop(:);
                    end
                end
            else
                if isempty(Sxs)
                    for t = 1:rB
                        deltaMloop = deltaM(:, t);
                        deltaSloop = deltaS(:, t);
                        if ~isempty(idx)
                            deltaMloop = repmat(deltaMloop(idx'), [1, 1, fi]);
                            deltaSloop = repmat(deltaSloop(idx'), [1, 1, fi]);
                        end
                        Cazloop = reshape(permute(reshape(Caz(:, t),...
                            [1, wi * hi, fi, B]), [1, 2, 4, 3]),...
                            [1, wi * hi * B, fi]);
                        Czz = Cazloop.*mw;
                        deltaMloop = sum(Czz .* deltaMloop, 1);
                        deltaSloop = sum(Czz .* deltaSloop .* Czz, 1);
                        deltaMloop = permute(reshape(deltaMloop,...
                            [wi * hi, B, fi]), [1, 3, 2]);
                        deltaSloop = permute(reshape(deltaSloop,...
                            [wi * hi, B, fi]), [1, 3, 2]);
                        deltaMz(:, t) = deltaMloop(:);
                        deltaSz(:, t) = deltaSloop(:);
                    end
                else
                    Caxs = bsxfun(@times, Sxs, J);
                    for t = 1:rB
                        deltaMloop = deltaM(:, t);
                        deltaSloop = deltaS(:, t);
                        if ~isempty(idx)
                            deltaMloop = repmat(deltaMloop(idx'), [1, 1, fi]);
                            deltaSloop = repmat(deltaSloop(idx'), [1, 1, fi]);
                        end
                        Cazloop  = reshape(permute(reshape(Caz(:, t),...
                            [1, wi*hi, fi, B]), [1, 2, 4, 3]),...
                            [1, wi * hi * B, fi]);
                        Caxsloop = reshape(permute(reshape(Caxs(:, t),...
                            [1, wi * hi, fi, B]), [1, 2, 4, 3]),...
                            [1, wi * hi * B, fi]);
                        Czz      = Cazloop .* mw;
                        Czx      = Caxsloop .* mw;
                        deltaMzloop    = sum(Czz .* deltaMloop, 1);
                        deltaSzloop    = sum(Czz .* deltaSloop .* Czz, 1);
                        deltaMzxloop   = sum(Czx .* deltaMloop, 1);
                        deltaSzxloop   = sum(Czx .* deltaSloop .* Czx, 1);
                        deltaMzloop    = permute(reshape(deltaMzloop,...
                            [wi * hi, B, fi]), [1, 3, 2]);
                        deltaSzloop    = permute(reshape(deltaSzloop,...
                            [wi * hi, B, fi]), [1, 3, 2]);
                        deltaMzxloop   = permute(reshape(deltaMzxloop,...
                            [wi * hi, B, fi]), [1, 3, 2]);
                        deltaSzxloop   = permute(reshape(deltaSzxloop,...
                            [wi * hi, B, fi]), [1, 3, 2]);
                        deltaMz(:, t)  = deltaMzloop(:);
                        deltaSz(:, t)  = deltaSzloop(:);
                        deltaMzx(:, t) = deltaMzxloop(:);
                        deltaSzx(:, t) = deltaSzxloop(:);
                    end
                end
            end
        end
        function [deltaMz, deltaSz,...
                deltaMzx, deltaSzx] = convHiddenStateBackwardPassB1(Sz, Sxs,...
                J, mw, deltaM, deltaS, idx, idxFCzwa, wi, hi, fi, gpu)
            n = size(idxFCzwa{1}, 1);
            mw = [mw; zeros(1, 1, 'like', mw)];
            mw = reshape(mw(idxFCzwa{1}), [n, wi * hi, fi]);                      
            if ~isempty(idx)
                deltaM = [deltaM; zeros(1, 1, 'like', deltaM)];
                deltaS = [deltaS; zeros(1, 1, 'like', deltaS)];
                deltaM = deltaM(idx)';
                deltaS = deltaS(idx)';
            end
            if isempty(Sxs)
                if gpu
                    Caz = bsxfun(@times, Sz, J);
                    Caz = reshape(Caz, [1, wi * hi, fi]);
                    [deltaMloop,...
                        deltaSloop] = arrayfun(@vectorizedDelta_V2, Caz,...
                        mw, deltaM, deltaS);
                else
                    Caz = Sz .* J;
                    Caz = reshape(Caz, [1, wi * hi, fi]);
                    [deltaMloop, deltaSloop] = vectorizedDelta_V2(Caz, mw,...
                        deltaM, deltaS);
                end
                deltaMloop = sum(deltaMloop, 1);
                deltaSloop = sum(deltaSloop, 1);
                deltaMz    = deltaMloop(:);
                deltaSz    = deltaSloop(:);
                deltaMzx   = Sxs;
                deltaSzx   = Sxs;
            else
                if gpu
                    Caz  = Sz .* J;
                    Caxs = Sxs .* J;
                    Caz  = reshape(Caz, [1, wi * hi, fi]);
                    Caxs = reshape(Caxs, [1, wi * hi, fi]);
                    [deltaMzloop, deltaSzloop, deltaMzxloop,...
                        deltaSzxloop] = arrayfun(@vectorized4delta, mw, Caz,...
                        Caxs, deltaM, deltaS);
                else
                    Caz = bsxfun(@times, Sz, J);
                    Caxs = bsxfun(@times, Sxs, J);
                    Caz = reshape(Caz, [1, wi * hi, fi]);
                    Caxs = reshape(Caxs, [1, wi * hi, fi]);
                    [deltaMzloop, deltaSzloop, deltaMzxloop,...
                        deltaSzxloop] = vectorized4delta(mw, Caz, Caxs,...
                        deltaM, deltaS);
                end
                deltaMzloop  = sum(deltaMzloop, 1);
                deltaSzloop  = sum(deltaSzloop, 1);
                deltaMzxloop = sum(deltaMzxloop, 1);
                deltaSzxloop = sum(deltaSzxloop, 1);
                deltaMz      = deltaMzloop(:);
                deltaSz      = deltaSzloop(:);
                deltaMzx     = deltaMzxloop(:);
                deltaSzx     = deltaSzxloop(:);
            end
        end
                          
        function [deltaMz, deltaSz, deltaMzx,...
                deltaSzx] = convHiddenStateBackwardPassMexcuda_V2(Sz, Sxs,...
                J, mw, deltaM, deltaS, idxFCzwa, idxSzzUd, wstarpos, wo, ho,...
                fo, wi, hi, fi, ki, B)                    
            [deltaMz, deltaSz] = convHiddenStateBackwardPass4matlab_V2(mw,...
                Sz, J, deltaM, deltaS, idxFCzwa{1}, idxSzzUd, wstarpos, wo,...
                ho, fo, wi, hi, fi, ki, B);
            if ~isempty(Sxs)
                [deltaMzx,...
                    deltaSzx] = convHiddenStateBackwardPass4matlab_V2(mw,...
                    Sxs, J, deltaM, deltaS, idxFCzwa{1}, idxSzzUd, wstarpos,...
                    wo, ho, fo, wi, hi, fi, ki, B);
            else
                deltaMzx = [];
                deltaSzx = [];
            end
        end
        function [deltaMw, deltaSw, deltaMb,...
                deltaSb] = convParameterBackwardPassMexcuda_V2(deltaMw,...
                deltaSw, deltaMb, deltaSb, Sw, Sb, ma, deltaM, deltaS,...
                idxFmwa, wstartpos, bstartpos, wo, ho, fo, wi, hi, fi, ki, B)  
            [deltaMw, deltaSw, deltaMb,...
                deltaSb] = convParamBackwardPass4matlab_V2(Sw, ma, Sb,...
                deltaM, deltaS, idxFmwa{2}, wstartpos, bstartpos, wo, ho,...
                fo, wi, hi, fi, ki, B, deltaMw, deltaSw, deltaMb, deltaSb);
        end
                       
        % Transposed convolutional layer        
        function [mz, Sz] = tconvMeanVar(mz, Sz, mw, Sw, mb, Sb, ma, Sa,...
                idxFmwa, wo, ho, fo, B, rB, gpu)           
            if any(~isnan(mb))
                mb = repmat(reshape(repmat(mb', [wo * ho, 1]),...
                    [wo * ho * fo, 1]), [B, 1]);
                Sb = repmat(reshape(repmat(Sb', [wo * ho, 1]),...
                    [wo * ho * fo, 1]), [B, 1]);
            else
                mb = zeros(1, 1, 'like', ma(1));
                Sb = zeros(1, 1, 'like', Sa(1));
            end
            n  = size(idxFmwa{1});
            mw = [mw; zeros(1, 1, 'like', mw)];
            Sw = [Sw; zeros(1, 1, 'like', Sw)];
            mw = repmat(mw(idxFmwa{1}), [1, 1, B]);  
            Sw = repmat(Sw(idxFmwa{1}), [1, 1, B]);
            ma = [ma; zeros(1, size(ma, 2), 'like', ma)];
            Sa = [Sa; zeros(1, size(ma, 2), 'like', Sa)];      
            if gpu
                for t = 1:rB
                    maloop = ma(:, t);
                    Saloop = Sa(:, t);
                    maloop = repmat(reshape(maloop(idxFmwa{2}),...
                        [n(1), wo * ho, B]), [1, fo, 1]);
                    Saloop = repmat(reshape(Saloop(idxFmwa{2}),...
                        [n(1), wo * ho, B]), [1, fo, 1]);
                    [mzloop, Szloop] = arrayfun(@vectorizedMeanVar, maloop,...
                        mw, Saloop, Sw);
                    mzloop = sum(mzloop, 1);
                    Szloop = sum(Szloop, 1);
                    mz(:, t) = mzloop(:);
                    Sz(:, t) = Szloop(:);
                end
                if any(~isnan(mb))
                    [mz, Sz] = arrayfun(@twoPlus, mz, Sz, mb, Sb);
                end
            else
                for t = 1:rB
                    maloop = ma(:, t);
                    Saloop = Sa(:, t);
                    maloop = repmat(reshape(maloop(idxFmwa{2}),...
                        [n(1), wo * ho, B]), [1, fo, 1]);
                    Saloop = repmat(reshape(Saloop(idxFmwa{2}),...
                        [n(1), wo * ho, B]), [1, fo, 1]);
                    [mzloop, Szloop] = vectorizedMeanVar(maloop, mw, Saloop,...
                        Sw);
                    mzloop = sum(mzloop, 1);
                    Szloop = sum(Szloop, 1);
                    mz(:, t) = mzloop(:) + mb;
                    Sz(:, t) = Szloop(:) + Sb;
                end
            end
        end
        function [deltaMw, deltaSw, deltaMb,...
                deltaSb] = tconvParameterBackwardPass(deltaMw, deltaSw,...
                deltaMb, deltaSb, Sw, Sb, ma, deltaM, deltaS, idx, idxFCwz,...
                ki, fi, wo, ho, fo, B, rB, gpu)  
            ma  = [ma; zeros(1, size(ma, 2), 'like', ma)]; 
            n   = size(idxFCwz{2});         
            Sw  = reshape(Sw, [1, ki * ki * fo, fi]);
            Cbz = repmat(Sb', [1, B]);                  
            if gpu  
                for t = 1:rB
                    maloop = ma(:, t);
                    maloop = repmat(reshape(maloop(idxFCwz{2}),...
                        [n(1), ki * ki, fi]), [1, fo, 1]);
                    deltaMwloop = [deltaM(:, t); zeros(1, 1, 'like', deltaM)];
                    deltaSwloop = [deltaS(:, t); zeros(1, 1, 'like', deltaM)];
                    [deltaMwloop,...
                        deltaSwloop] = arrayfun(@vectorizedDelta_V2, Sw,...
                        maloop, deltaMwloop(idx), deltaSwloop(idx));
                    deltaMwloop = sum(deltaMwloop, 1);
                    deltaSwloop = sum(deltaSwloop, 1);
                    deltaMw(:, t) = deltaMwloop(:);
                    deltaSw(:, t) = deltaSwloop(:);                  
                end
                if any(~isnan(Sb))
                    for t = 1:rB
                        deltaMbloop = reshape(deltaM(:, t), [ho * wo, fo * B]);
                        deltaSbloop = reshape(deltaS(:, t), [ho * wo, fo * B]);
                        [deltaMbloop,...
                            deltaSbloop] = arrayfun(@vectorizedDelta, Cbz,...
                            deltaMbloop, deltaSbloop);
                        deltaMb(:, t) = sum(reshape(sum(deltaMbloop, 1),...
                            [fo, B]), 2);
                        deltaSb(:, t) = sum(reshape(sum(deltaSbloop, 1),...
                            [fo, B]), 2);
                    end
                end
            else 
                for t = 1:rB
                    maloop = ma(:, t);
                    maloop = repmat(reshape(maloop(idxFCwz{2}),...
                        [n(1), ki * ki, fi]), [1, fo, 1]);
                    deltaMwloop = [deltaM(:, t); zeros(1, 1, 'like', deltaM)];
                    deltaSwloop = [deltaS(:, t); zeros(1, 1, 'like', deltaM)];
                    Cwz = Sw .* maloop;        
                    deltaMwloop = Cwz .* deltaMwloop;
                    deltaSwloop = Cwz .* deltaSwloop .* Cwz;
                    deltaMwloop = sum(deltaMwloop, 1);
                    deltaSwloop = sum(deltaSwloop, 1);
                    deltaMw(:, t) = deltaMwloop(:);
                    deltaSw(:, t) = deltaSwloop(:);
                    if any(~isnan(Sb))
                        deltaMbloop = reshape(deltaM(:, t), [ho * wo, fo * B]);
                        deltaSbloop = reshape(deltaS(:, t), [ho * wo, fo * B]);
                        deltaMbloop = Cbz .* deltaMbloop;
                        deltaSbloop = Cbz .* deltaSbloop .* Cbz;
                        deltaMb(:, t) = sum(reshape(sum(deltaMbloop, 1), [fo, B]), 2);
                        deltaSb(:, t) = sum(reshape(sum(deltaSbloop, 1), [fo, B]), 2);
                    end
                end
            end
            deltaMw = sum(deltaMw, 2);
            deltaSw = sum(deltaSw, 2);
            deltaMb = sum(deltaMb, 2);
            deltaSb = sum(deltaSb, 2);
        end
        function [deltaMz, deltaSz, deltaMxs,...
                deltaSxs] = tconvHiddenStateBackwardPass(Sz, Sxs, J, mw,...
                deltaM, deltaS, idx, idxFCzwa, wi, hi, fi, B, rB, gpu)
            deltaMz  = Sz;
            deltaSz  = Sz;
            deltaMxs = Sxs;
            deltaSxs = Sxs;
            mw       = [mw; zeros(1, 1, 'like', mw)];
            mw       = repmat(mw(idxFCzwa{1}), [1, 1, B]);    
            n        = size(idx); 
            Caz      = bsxfun(@times, J, Sz);             
            deltaM   = [deltaM; zeros(1, size(deltaM, 2), 'like', deltaM)];
            deltaS   = [deltaS; zeros(1, size(deltaS, 2), 'like', deltaS)];
            if gpu
                if isempty(Sxs)
                    for t = 1:rB
                        deltaMzloop = deltaM(:, t);
                        deltaSzloop = deltaS(:, t);
                        deltaMzloop = repmat(reshape(deltaMzloop(idx),...
                            [n(1), wi * hi, B]), [1, fi, 1]);
                        deltaSzloop = repmat(reshape(deltaSzloop(idx),...
                            [n(1), wi * hi, B]), [1, fi, 1]);
                        Cazloop     = reshape(Caz(:, t), [1, wi * hi * fi, B]);
                        
                        [deltaMzloop,...
                            deltaSzloop] = arrayfun(@vectorizedDelta_V2, mw,...
                            Cazloop, deltaMzloop, deltaSzloop);
                        deltaMzloop = sum(deltaMzloop, 1);
                        deltaSzloop = sum(deltaSzloop, 1);
                        deltaMz(:, t) = deltaMzloop(:);
                        deltaSz(:, t) = deltaSzloop(:);
                    end
                else
                    Caxs = bsxfun(@times, J, Sxs);   
                    for t = 1:rB
                        deltaMloop = deltaM(:, t);
                        deltaSloop = deltaS(:, t);
                        deltaMloop = repmat(reshape(deltaMloop(idx),...
                            [n(1), wi * hi, B]), [1, fi, 1]);
                        deltaSloop = repmat(reshape(deltaSloop(idx),...
                            [n(1), wi * hi, B]), [1, fi, 1]);
                        Cazloop    = reshape(Caz(:, t), [1, wi * hi * fi, B]);
                        Caxsloop   = reshape(Caxs(:, t), [1, wi * hi * fi, B]);
                        [deltaMzloop, deltaSzloop, deltaMxsloop,...
                            deltaSxsloop] = arrayfun(@vectorized4delta, mw,...
                            Cazloop, Caxsloop, deltaMloop, deltaSloop);
                        
                        deltaMzloop    = sum(deltaMzloop, 1);
                        deltaSzloop    = sum(deltaSzloop, 1);
                        deltaMxsloop   = sum(deltaMxsloop, 1);
                        deltaSxsloop   = sum(deltaSxsloop, 1);
                        deltaMz(:, t)  = deltaMzloop(:);
                        deltaSz(:, t)  = deltaSzloop(:);
                        deltaMxs(:, t) = deltaMxsloop(:);
                        deltaSxs(:, t) = deltaSxsloop(:);
                    end
                end
            else
                if isempty(Sxs)
                    for t = 1:rB
                        deltaMzloop = deltaM(:, t);
                        deltaSzloop = deltaS(:, t);
                        deltaMzloop = repmat(reshape(deltaMzloop(idx),...
                            [n(1), wi * hi, B]), [1, fi, 1]);
                        deltaSzloop = repmat(reshape(deltaSzloop(idx),...
                            [n(1), wi * hi, B]), [1, fi, 1]);
                        Cazloop     = reshape(Caz(:, t), [1, wi * hi * fi, B]);
                        deltaMzloop = sum(mw .* Cazloop .* deltaMzloop, 1);
                        deltaSzloop = sum(mw .* Cazloop .* deltaSzloop .* mw .* Cazloop, 1);
                        deltaMz(:, t) = deltaMzloop(:);
                        deltaSz(:, t) = deltaSzloop(:);
                    end
                else
                    Caxs = bsxfun(@times, J, Sxs);
                    for t = 1:rB
                        deltaMloop = deltaM(:, t);
                        deltaSloop = deltaS(:, t);
                        deltaMloop = repmat(reshape(deltaMloop(idx),...
                            [n(1), wi * hi, B]), [1, fi, 1]);
                        deltaSloop = repmat(reshape(deltaSloop(idx),...
                            [n(1), wi * hi, B]), [1, fi, 1]);
                        Cazloop    = reshape(Caz(:, t), [1, wi * hi * fi, B]);
                        Caxsloop   = reshape(Caxs(:, t), [1, wi * hi * fi, B]);                       
                        deltaMzloop    = sum(mw .* Cazloop .* deltaMloop, 1);
                        deltaSzloop    = sum(mw .* Cazloop .* deltaSloop .* mw .* Cazloop, 1);
                        deltaMxsloop   = sum(mw .* Caxsloop .* deltaMloop, 1);
                        deltaSxsloop   = sum(mw .* Caxsloop .* deltaSloop .* mw .* Caxsloop, 1);
                        deltaMz(:, t)  = deltaMzloop(:);
                        deltaSz(:, t)  = deltaSzloop(:);
                        deltaMxs(:, t) = deltaMxsloop(:);
                        deltaSxs(:, t) = deltaSxsloop(:);
                    end
                end
            end
        end        
        
        % Shortcut for residual network
        function [mz, Sz] = xshortcutMeanVar(mz, Sz, mw, Sw, ma, Sa,...
                idxFmwa, wo, ho, fi, fo, k, B, rB, padding, gpu)           
            mb = zeros(1, 1, 'like', ma(1));
            Sb = zeros(1, 1, 'like', Sa(1));
            mw = reshape(repmat(reshape(mw, [k * k * fi, fo]),...
                [B * wo * ho, 1]), [k * k * fi, fo * B * wo * wo]);  
            Sw = reshape(repmat(reshape(Sw, [k * k * fi, fo]),...
                [B * wo * ho, 1]), [k * k * fi, fo * B * wo * wo]); 
            if padding ~= 0 
                ma = [ma; zeros(1, size(ma, 2), 'like', ma)];
                Sa = [Sa; zeros(1, size(Sa, 2), 'like', Sa)];
            end
            for t = 1:rB
                maloop = ma(:, t);
                Saloop = Sa(:, t);
                maloop = repmat(maloop(idxFmwa{2}), [1, fo]);
                Saloop = repmat(Saloop(idxFmwa{2}), [1, fo]);
                if gpu
                    [mzloop, Szloop] = arrayfun(@vectorizedMeanVar, maloop,...
                        mw, Saloop, Sw);
                    mzloop = bsxfun(@plus, sum(mzloop, 1), mb);
                    Szloop = bsxfun(@plus, sum(Szloop, 1), Sb);
                else
                    [mzloop, Szloop] = vectorizedMeanVar(maloop, mw, Saloop,...
                        Sw);
                    mzloop = sum(mzloop, 1) + mb;
                    Szloop = sum(Szloop, 1) + Sb;
                end
                mz(:, t) = reshape(permute(reshape(mzloop, [wo, ho, B, fo]),...
                    [1 2 4 3]), [wo * ho * fo * B, 1]);
                Sz(:, t) = reshape(permute(reshape(Szloop, [wo, ho, B, fo]),...
                    [1 2 4 3]), [wo * ho * fo * B, 1]);
            end
        end
        function [deltaMxs, deltaSxs,...
                deltaMdxs, deltaSdxs] = xshortDelta(deltaM, deltaS, Sxs,...
                Sdxs, J, mwx, idx, idxFCzwaXsc, fi, B, rB, q, gpu)           
            if ~isempty(idx)
                deltaMxs  = zeros(size(Sxs), 'like', Sxs);
                deltaSxs  = deltaMxs;
                deltaMdxs = deltaMxs;
                deltaSdxs = deltaMxs;
                B    = cast(B, 'like', deltaM);
                fi   = cast(fi, 'like', deltaM);
                wh   = sqrt(cast(q / (fi * B), 'like', deltaM));
                
                n    = size(idxFCzwaXsc{1}, 2);
                q2   = size(idxFCzwaXsc{1}, 1);
                wh2  = sqrt(q2 / fi);
                mwx  = [mwx ; zeros(1, 1, 'like', mwx)];
                mwx  = reshape(repmat(reshape(mwx(idxFCzwaXsc{1}),...
                    [wh2 * wh2, n, fi]), [B, 1, 1]), [q2 * B, n]);
                Cax  = bsxfun(@times, J, Sxs);
                Cadx = bsxfun(@times, J, Sdxs);
                if gpu
                    for t = 1:rB
                        Caxloop  = Cax(:, t);
                        Cadxloop = Cadx(:, t);
                        Caxloop  = reshape(permute(reshape(Caxloop(idxFCzwaXsc{2}),...
                            [wh2, wh2, fi, B]), [1, 2, 4, 3]), [q2 * B, 1]);
                        Cadxloop = reshape(permute(reshape(Cadxloop(idxFCzwaXsc{2}),...
                            [wh2, wh2, fi, B]), [1, 2, 4, 3]), [q2 * B, 1]);
                        
                        deltaMloop = deltaM(:, t);
                        deltaSloop = deltaS(:, t);
                        deltaMloop = repmat(deltaMloop(idx), [fi, 1]);
                        deltaSloop = repmat(deltaSloop(idx), [fi, 1]);
                        [deltaMxsloop, deltaSxsloop, deltaMdxsloop,...
                            deltaSdxsloop] = arrayfun(@vectorized4delta, mwx,...
                            Caxloop, Cadxloop, deltaMloop, deltaSloop);
                        deltaMxs(idxFCzwaXsc{2}, t)  = reshape(permute(reshape(sum(deltaMxsloop, 2),...
                            [wh, wh, B, fi]), [1, 2, 4, 3]), [wh * wh * fi * B, 1]);
                        deltaSxs(idxFCzwaXsc{2}, t)  = reshape(permute(reshape(sum(deltaSxsloop, 2),...
                            [wh, wh, B, fi]), [1, 2, 4, 3]), [wh * wh * fi * B, 1]);
                        deltaMdxs(idxFCzwaXsc{2}, t) = reshape(permute(reshape(sum(deltaMdxsloop, 2),...
                            [wh, wh, B, fi]), [1, 2, 4, 3]), [wh * wh * fi * B, 1]);
                        deltaSdxs(idxFCzwaXsc{2}, t) = reshape(permute(reshape(sum(deltaSdxsloop, 2),...
                            [wh, wh, B, fi]), [1, 2, 4, 3]), [wh * wh * fi * B, 1]);
                    end
                else
                    for t = 1:rB
                        Caxloop  = Cax(:, t);
                        Cadxloop = Cadx(:, t);
                        Caxloop  = reshape(permute(reshape(Caxloop(idxFCzwaXsc{2}),...
                            [wh2, wh2, fi, B]), [1, 2, 4, 3]), [q2 * B, 1]);
                        Cadxloop = reshape(permute(reshape(Cadxloop(idxFCzwaXsc{2}),...
                            [wh2, wh2, fi, B]), [1, 2, 4, 3]), [q2 * B, 1]);
                        
                        deltaMloop = deltaM(:, t);
                        deltaSloop = deltaS(:, t);
                        deltaMloop = repmat(deltaMloop(idx), [fi, 1]);
                        deltaSloop = repmat(deltaSloop(idx), [fi, 1]);
                        deltaMxsloop  = sum(Caxloop .* deltaMloop, 2);
                        deltaSxsloop  = sum((Caxloop.^2) .* deltaSloop, 2);
                        deltaMdxsloop = sum(Cadxloop .* deltaMloop, 2);
                        deltaSdxsloop = sum((Cadxloop.^2) .* deltaSloop, 2);
                        deltaMxs(idxFCzwaXsc{2}, t)  = reshape(permute(reshape(deltaMxsloop,...
                            [wh, wh, B, fi]), [1, 2, 4, 3]), [wh * wh * fi * B, 1]);
                        deltaSxs(idxFCzwaXsc{2}, t)  = reshape(permute(reshape(deltaSxsloop,...
                            [wh, wh, B, fi]), [1, 2, 4, 3]), [wh * wh * fi * B, 1]);                        
                        deltaMdxs(idxFCzwaXsc{2}, t) = reshape(permute(reshape(deltaMdxsloop,...
                            [wh, wh, B, fi]), [1, 2, 4, 3]), [wh * wh * fi * B, 1]);
                        deltaSdxs(idxFCzwaXsc{2}, t) = reshape(permute(reshape(deltaSdxsloop,...
                            [wh, wh, B, fi]), [1, 2, 4, 3]), [wh * wh * fi * B, 1]);
                    end
                end
            else               
                if gpu                  
                    if ~isempty(Sxs)
                        [deltaMxs, deltaSxs, deltaMdxs,...
                            deltaSdxs] = arrayfun(@vectorized4delta, J, Sxs,...
                            Sdxs, deltaM, deltaS);
                    else
                        [deltaMdxs,...
                            deltaSdxs] = arrayfun(@vectorizedDelta_V2, J,...
                            Sdxs, deltaM, deltaS);
                        deltaMxs = [];
                        deltaSxs = [];
                    end                   
                else
                    if ~isempty(Sxs)
                        Cxx = J .* Sxs;
                        deltaMxs = Cxx .* deltaM;
                        deltaSxs = (Cxx.^2) .* deltaS;
                    else
                        deltaMxs = [];
                        deltaSxs = [];
                    end
                    Cdxx = J .* Sdxs;
                    deltaMdxs = Cdxx .* deltaM;
                    deltaSdxs = (Cdxx.^2) .* deltaS;
                end                
            end          
        end         
        function [deltaMxs, deltaSxs, deltaMdxs,...
                deltaSdxs] = convscHiddenStateBackwardPassMexcuda(mwx, J,...
                Sxs, Sdxs, deltaM, deltaS, wstartpos, idxFCzwa, idxSzzUd,...
                wo, ho, fo, wi, hi, fi, ki, B) 
            n    = size(idxFCzwa{1}, 1) * fo;
            q2   = size(idxFCzwa{1}, 2);
            wh2  = sqrt(q2);           
            [deltaMxs,...
             deltaSxs]   = convscHiddenStateBackwardPass4matlab_V2(mwx, Sxs,...
             J, deltaM, deltaS, wstartpos, wo*ho, fo, wi * hi, wh2 * wh2,...
             fi, ki * ki, n, B, wi * hi * fi * B, idxFCzwa{1}, idxSzzUd,...
             idxFCzwa{2});
            
            [deltaMdxs,...
             deltaSdxs] = convscHiddenStateBackwardPass4matlab_V2(mwx, Sdxs,...
             J, deltaM, deltaS, wstartpos, wo * ho, fo, wi * hi, wh2 * wh2,...
             fi, ki * ki, n, B, wi * hi * fi * B, idxFCzwa{1}, idxSzzUd,...
             idxFCzwa{2});
        end
        function [deltaMxs, deltaSxs] = scHiddenStateBackwardPass(Sxs, J,...
                deltaM, deltaS) 
            [deltaMxs, deltaSxs] = scHiddenStateBackwardPass4matlab(Sxs, J,...
                deltaM, deltaS);
        end                  
        
        % Shared functions for update step
        function [deltaM, deltaS] = inovationVector(SzF, dMz, dSz, gpu)
            if gpu
                iSzF  = bsxfun(@rdivide, 1, SzF);
                iSzF(isinf(iSzF)) = zeros(1,1, 'like', dMz);
                [deltaM, deltaS] = arrayfun(@vectorizedDelta, iSzF, dMz, dSz);
            else              
                iSzF   = 1./SzF; 
                iSzF(isinf(iSzF)) = zeros(1,1, 'like', dMz);
                [deltaM, deltaS]  = vectorizedDelta(iSzF, dMz, dSz);
            end           
        end 
        function [deltaMz, deltaSz] = fowardHiddenStateUpdate(mzF, SzF, Cyz,...
                y, gpu)
            if gpu
                dz  = y - mzF;
                SzF = 1 ./ SzF;
                SzF(isinf(SzF)) = 0;
                K = bsxfun(@times, Cyz, SzF);
                deltaMz = bsxfun(@times, K, dz);
                deltaSz = bsxfun(@times, -K, Cyz);
            else
                dz  = y - mzF;
                SzF = 1 ./ SzF;
                SzF(isinf(SzF)) = 0;
                K = Cyz .* SzF;
                deltaMz = K .* dz;
                deltaSz = -K .* Cyz;
            end
        end   
        function theta = globalParameterUpdate(theta, deltaTheta, gpu)          
            [mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx] = tagi.extractParameters(theta);
            [deltaMw, deltaSw, deltaMb, deltaSb, deltaMwx, deltaSwx,...
                deltaMbx, deltaSbx] = tagi.extractParameters(deltaTheta);
            if gpu
                [mw, Sw]   = arrayfun(@twoPlus, mw, Sw, deltaMw, deltaSw);
                [mb, Sb]   = arrayfun(@twoPlus, mb, Sb, deltaMb, deltaSb);
                [mwx, Swx] = arrayfun(@twoPlus, mwx, Swx, deltaMwx, deltaSwx);
                [mbx, Sbx] = arrayfun(@twoPlus, mbx, Sbx, deltaMbx, deltaSbx);
            else
                [mw, Sw]   = twoPlus(mw, Sw, deltaMw, deltaSw);
                [mb, Sb]   = twoPlus(mb, Sb, deltaMb, deltaSb);
                [mwx, Swx] = twoPlus(mwx, Swx, deltaMwx, deltaSwx);
                [mbx, Sbx] = twoPlus(mbx, Sbx, deltaMbx, deltaSbx);
            end
            theta = tagi.compressParameters(mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx);
        end
        function theta = globalParameterUpdateMultiGPUs(theta, deltaTheta,...
                numParamsPerlayer, numDevices)  
            numParams  = sum(numParamsPerlayer, 2);           
            deltaTheta = cat(2, deltaTheta{:});
            [deltaMw, deltaSw, deltaMb, deltaSb, deltaMwx, deltaSwx,...
                deltaMbx, deltaSbx] = tagi.extractParameters_V2(deltaTheta);
            deltaMw  = cat(1, deltaMw{:});
            deltaSw  = cat(1, deltaSw{:});
            deltaMb  = cat(1, deltaMb{:});
            deltaSb  = cat(1, deltaSb{:});
            deltaMwx = cat(1, deltaMwx{:});
            deltaSwx = cat(1, deltaSwx{:});
            deltaMbx = cat(1, deltaMbx{:});
            deltaSbx = cat(1, deltaSbx{:});  
            
            deltaMw  = sum(reshape(cat(1, deltaMw{:}), [numParams(1), numDevices]), 2);
            deltaSw  = sum(reshape(cat(1, deltaSw{:}), [numParams(1), numDevices]), 2);
            deltaMb  = sum(reshape(cat(1, deltaMb{:}), [numParams(2), numDevices]), 2);
            deltaSb  = sum(reshape(cat(1, deltaSb{:}), [numParams(2), numDevices]), 2);
            deltaMwx = sum(reshape(cat(1, deltaMwx{:}), [numParams(3), numDevices]), 2);
            deltaSwx = sum(reshape(cat(1, deltaSwx{:}), [numParams(3), numDevices]), 2);
            deltaMbx = sum(reshape(cat(1, deltaMbx{:}), [numParams(4), numDevices]), 2);
            deltaSbx = sum(reshape(cat(1, deltaSbx{:}), [numParams(4), numDevices]), 2);            
            [mw, Sw, mb, Sb, mwx, Swx,...
                mbx, Sbx] = tagi.extractParameters(theta);
            [mw, Sw, mb, Sb, mwx, Swx,...
                mbx, Sbx] = tagi.catParameters(mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx);            
            [mw, Sw]   = arrayfun(@twoPlus, mw, Sw, deltaMw, deltaSw);
            [mb, Sb]   = arrayfun(@twoPlus, mb, Sb, deltaMb, deltaSb);
            [mwx, Swx] = arrayfun(@twoPlus, mwx, Swx, deltaMwx, deltaSwx);
            [mbx, Sbx] = arrayfun(@twoPlus, mbx, Sbx, deltaMbx, deltaSbx);
            [mw, Sw, mb, Sb, mwx, Swx,...
                mbx, Sbx] = tagi.distributeParameters2Layers(mw, Sw, mb, Sb,...
                mwx, Swx, mbx, Sbx, numParamsPerlayer);
            theta = tagi.compressParameters(mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx);
        end
        function [mnorm, Snorm] = distrNorm(m, S)
            N     = length(S);
            mhat  = mean(m);
            Shat  = 1 / N * (sum(S, 1)+ sum((m - mhat).^2, 1)) + 1E-8;
            mnorm = (m - mhat) ./ (sqrt(abs(Shat)));
            Snorm = (1 ./ abs(Shat)) .* S;
        end         
        function theta = globalParameterUpdateCUDA(theta, deltaTheta,...
                wxupdate, numParams)          
            [mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx] = tagi.extractParameters(theta);
            [deltaMw, deltaSw, deltaMb, deltaSb, deltaMwx, deltaSwx,...
                deltaMbx, deltaSbx] = tagi.extractParameters(deltaTheta);
            
            [mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx] = globalParamUpdate(mw, Sw,...
                mb, Sb, mwx, Swx, mbx, Sbx, deltaMw, deltaSw, deltaMb, deltaSb,...
                deltaMwx, deltaSwx, deltaMbx, deltaSbx, numParams(1, end),...
                numParams(2, end), numParams(3, end), numParams(4, end), wxupdate);
            
            theta = tagi.compressParameters(mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx);
        end
        
        % Noise update
        function [l, v2] = detachMeanVar(x, nl, nv2, B, rB)
            x  = reshape(x, [nl + nv2, B * rB]);
            l  = reshape(x(1 : nl, :), [B * nl, rB]);
            v2 = reshape(x(nl + 1 : end, :), [B * nv2, rB]);
        end
        function x = attachMeanVar(l, v2, nl, nv2, B, rB)
            l  = reshape(l, [nl, B * rB]);
            v2 = reshape(v2, [nv2, B * rB]);
            x  = [l; v2];
            x  = reshape(x, [(nl + nv2) * B, rB]);
        end
        function [deltaMlz, deltaSlz, deltaMv2z,...
                deltaSv2z] = noiseUpdate4classification(Slz, mla, Sla, J,...
                Jv2, mv2a, Sv2a, Cv2a, y, sv, udIdx, gpu)
            Cyz = J(udIdx) .* Slz(udIdx) ;
            Syf = Sla(udIdx) + mv2a(udIdx) + Sv2a(udIdx) ./ (4 * mv2a(udIdx))...
                + sv.^2;
            [deltaMlz, deltaSlz] = tagi.fowardHiddenStateUpdate(mla(udIdx),...
                Syf, Cyz, y(udIdx), gpu);
            [deltaMv, deltaSv] = tagi.fowardHiddenStateUpdate(mla(udIdx),...
                Syf, mv2a(udIdx) + Sv2a(udIdx)./(4*mv2a(udIdx)), y(udIdx), gpu);
            mvUd = deltaMv;
            SvUd = mv2a(udIdx) + Sv2a(udIdx) ./ (4 * mv2a(udIdx)) + deltaSv;
            % Update activated standard deviation for z
            yv2  = mvUd.^2 + SvUd;
            Sv2f = 2 * SvUd.^2 + 4 * (mvUd.^2) .* SvUd;   
            [deltaMv2z, deltaSv2z] = tagi.noiseBackwardUpdate(mv2a(udIdx),...
                3 * Sv2a(udIdx) + 2 * mv2a(udIdx).^2,...
                Jv2(udIdx) .* Cv2a(udIdx), yv2, Sv2f, gpu); 
        end
        function [deltaMlz, deltaSlz, deltaMv2z,...
                deltaSv2z] = noiseUpdate4regression(Slz, mla, Sla, J, Jv2,...
                mv2a, Sv2a, Cv2a, y, sv, gpu)
            Cyz = J .* Slz ;
            Syf = Sla + mv2a + sv.^2;
            
            [deltaMlz, deltaSlz] = tagi.fowardHiddenStateUpdate(mla, Syf,...
                Cyz, y, gpu);
            [deltaMv, deltaSv] = tagi.fowardHiddenStateUpdate(mla, Syf,...
                mv2a, y, gpu);
            
            mvUd = deltaMv;
            SvUd = mv2a + deltaSv;
            
            % Update activated standard deviation for z
            yv2  = mvUd.^2 + SvUd;
            Sv2f = 2 * SvUd.^2 + 4 * (mvUd.^2) .* SvUd;   
            [deltaMv2z, deltaSv2z] = tagi.noiseBackwardUpdate(mv2a,...
                3 * Sv2a + 2 * mv2a.^2, Jv2 .* Cv2a, yv2, Sv2f, gpu); 
        end
        function [deltaMlz, deltaSlz, deltaMv2z,...
                deltaSv2z] = homoNoiseUpdate4regression(Slz, mla, Sla, J,...
                mv2a, Sv2a, y, gpu)
            Cyz = J .* Slz ;
            Syf = Sla + mv2a;
            
            [deltaMlz, deltaSlz] = tagi.fowardHiddenStateUpdate(mla, Syf,...
                Cyz, y, gpu);
            [deltaMv, deltaSv] = tagi.fowardHiddenStateUpdate(mla, Syf,...
                mv2a, y, gpu);
            
            mvUd = deltaMv;
            SvUd = mv2a + deltaSv;
            
            % Update activated standard deviation for z
            yv2  = mvUd.^2 + SvUd;
            Sv2f = 2 * SvUd.^2 + 4 * (mvUd.^2) .* SvUd; 
            Cv2a = Sv2a;
            [deltaMv2z, deltaSv2z] = tagi.noiseBackwardUpdate(mv2a,...
                3 * Sv2a + 2 * mv2a.^2, Cv2a, yv2, Sv2f, gpu); 
        end
        function [mlz, Slz, mv2z, Sv2z] = noiseUpdate4decoding(mlz, Slz,...
                mla, Sla, J, mv2z, Sv2z, Jv2, mv2a, Sv2a, Cv2a, y, sv, nl,...
                nv2, B, gpu)
            if nl~=nv2
                mv2aM = reshape(repmat(mv2a', [nl, 1]), [B*nl, 1]);
            else
                mv2aM = mv2a;
            end
            Cyz   = J .* Slz ;
            Syf   = Sla + mv2aM + sv.^2;
            [mlz, Slz] = tagi.fowardHiddenStateUpdate(mlz, Slz, mla, Syf,...
                Cyz, y, gpu);
            [deltaM, deltaS] = tagi.forwardNoiseUpdate(mla, Syf, mv2aM, y,...
                gpu);
            if nl~=nv2
                deltaM = sum(reshape(deltaM, [nl, B]), 1);
                deltaS = sum(reshape(deltaS, [nl, B]), 1);
                mvUd   = deltaM';
                SvUd   = mv2a + deltaS';
            else
                mvUd   = deltaM;
                SvUd   = mv2a + deltaS;
            end
            % Update activated standard deviation for z
            yv2  = mvUd.^2 + SvUd;
            Sv2f = Sv2a + 2 * mv2a.^2 + 2 * SvUd.^2 + 4 * (mvUd.^2) .* SvUd;   
            [mv2z, Sv2z] = tagi.noiseBackwardUpdate(mv2z, Sv2z, mv2a,...
                Sv2a + 2 * mv2a.^2, Jv2 .* Cv2a, yv2, Sv2f, gpu);
        end
        function [deltaMlz, deltaSlz, deltaMv2z,...
                deltaSv2z] = noiseUpdate4encoding(Slz, mla, Sla, J, Jv2,...
                mv2a, Sv2a, Cv2a, y, Sy, gpu)
            Cyz = J.*Slz ;
            [deltaMlz, deltaSlz] = tagi.noiseBackwardUpdate_V2(mla,...
                Sla + mv2a, Cyz, y, Sy, gpu);
            [deltaMv, deltaSv] = tagi.noiseBackwardUpdate(mla, Sla + mv2a,...
                mv2a, y, Sy, gpu);
            mvUd = deltaMv;
            SvUd = mv2a + deltaSv;
            % Update activated standard deviation for z
            yv2  = mvUd.^2 + SvUd;
            Sv2f = Sv2a +2 * SvUd.^2 + 4 * (mvUd.^2) .* SvUd;   
            [deltaMv2z, deltaSv2z] = tagi.noiseBackwardUpdate_V2(mv2a,...
                Sv2a + 2 * mv2a.^2, Jv2 .* Cv2a, yv2, Sv2f, gpu);
        end    
        function [deltaMz, deltaSz] = noiseBackwardUpdate(maF, SaF, CzzF,...
                maB, SaB, gpu)            
            if gpu
                funM    = @(x, y, z) x .* (y - z);
                funS    = @(x, y, z) x .* (y - z) .* x;
                Jz      = CzzF ./ SaF; 
                deltaMz = arrayfun(funM, Jz, maB, maF);
                deltaSz = arrayfun(funS, Jz, SaB, SaF);
            else
                Jz      = CzzF ./ SaF; 
                deltaMz = Jz .* (maB - maF);
                deltaSz = Jz .* (SaB - SaF) .* Jz;
            end
        end  
        
        % Initialization for weights and bias   
        function theta = initializeWeightBias(net)
            %Initialization
            nodes     = double(net.nodes);
            numLayers = length(net.nodes);
            layer     = net.layer;
            idxw      = net.idxw;
            idxwXsc   = net.idxwXsc;
            idxbXsc   = net.idxbXsc;
            idxb      = net.idxb;
            biasStd   = 1E-2;
            gainMw    = cast(net.gainMw, net.dtype);
            gainSw    = cast(net.gainSw, net.dtype); 
            gainMb    = cast(net.gainMb, net.dtype);
            gainSb    = cast(net.gainSb, net.dtype); 
            mw        = tagi.createInitCellwithArray(numLayers-1, net.dtype,...
                net.gpu);
            Sw        = mw;
            mb        = mw;
            Sb        = mw;
            mwx       = mw;
            Swx       = mw;
            mbx       = mw;
            Sbx       = mw;
            for j = 2:numLayers
                if ~isempty(idxw{j-1})                    
                    if layer(j) == net.layerEncoder.conv ...
                            || layer(j) == net.layerEncoder.tconv % Conv. layer
                        fanIn  = (cast(net.kernelSize(j-1), net.dtype) .^2) ...
                            * cast(net.filter(j-1), net.dtype);
                        if net.xsc(j-1)~=0
                            fanIn = 2 * fanIn;
                        end
                        if strcmp(net.initParamType, 'Xavier')
                            if j < numLayers ...
                                    && (layer(j+1) == net.layerEncoder.mp ...
                                    || layer(j+1) == net.layerEncoder.ap)
                                fanOut = ((cast(net.kernelSize(j-1), net.dtype) .^ 2) ...
                                    * cast(net.filter(j), net.dtype)) ...
                                    / (cast(net.kernelSize(j), net.dtype) .^2);
                            else
                                fanOut = ((cast(net.kernelSize(j-1), net.dtype).^2) ...
                                    * cast(net.filter(j), net.dtype));
                            end
                            scale = 2 / (fanIn + fanOut);
                            Sw{j-1} = (gainSw(j-1)) * scale * ...
                                ones(length(idxw{j-1}), 1, net.dtype);
                        elseif strcmp(net.initParamType, 'He')
                            scale   = 1 / (fanIn);
                            Sw{j-1} = (gainSw(j-1)) * scale * ...
                                ones(length(idxw{j-1}), 1, net.dtype);
                        end 
                        mw{j-1} = gainMw(j-1) * randn(length(Sw{j-1}), 1) ...
                            .* sqrt(Sw{j-1});
                        if ~isempty(idxb{j-1})
                            Sb{j-1} = gainSb(j-1) * biasStd * ...
                                ones(length(idxb{j-1}), 1, net.dtype);
                            mb{j-1} = gainMb(j-1) * randn(length(Sb{j-1}), 1) ...
                                .* sqrt(Sb{j-1});
                        end
                    elseif layer(j) == net.layerEncoder.ln ...
                            || layer(j) == net.layerEncoder.bn
                        Sb{j-1} = 1E-4 * gainSw(j-1) * ...
                            ones(length(idxb{j-1}), 1, net.dtype);
                        mb{j-1} = 0 * rand(length(Sb{j-1}), 1, net.dtype) ...
                            .* sqrt(Sb{j-1});
                        Sw{j-1} = 1 * ones(length(idxw{j-1}), 1, net.dtype);
                        mw{j-1} = 1 * ones(length(idxw{j-1}), 1, net.dtype);
                    else
                        fanIn  = nodes(j-1);
                        fanOut = nodes(j);
                        if strcmp(net.initParamType, 'Xavier')
                            scale = 2/(fanIn + fanOut);
                            Sw{j-1} = (gainSw(j-1)) * scale * ...
                                ones(length(idxw{j-1}), 1, net.dtype);
                        elseif strcmp(net.initParamType, 'He')
                            scale = 1 / fanIn;
                            Sw{j-1} = (gainSw(j-1)) * scale * ...
                                ones(length(idxw{j-1}), 1, net.dtype);
                        end
                        mw{j-1} = gainMw(j-1)*randn(length(Sw{j-1}), 1) ...
                            .* sqrt(Sw{j-1});
                        if ~isempty(idxb{j-1})
                            Sb{j-1} = gainSb(j-1) * scale * ...
                                ones(length(idxb{j-1}), 1, net.dtype);
                            mb{j-1} = gainSb(j-1) * randn(length(Sb{j-1}), 1) ...
                                .* sqrt(Sb{j-1});
                        end
                    end  
                end 
                if net.xsc(j) ~=0 ...
                        && (net.filter(net.xsc(j)) ~= net.filter(j) ...
                        || net.imgW(net.xsc(j)) ~= net.imgW(j))
                    idxXsc = net.xsc(j);                                     
                    fanIn  = cast(net.filter(idxXsc), net.dtype);
                    fanOut = cast(net.filter(j), net.dtype);
                    if strcmp(net.initParamType, 'Xavier')
                        Swx{idxXsc} = (gainS(idxXsc)) * (2 / (fanIn + fanOut)) ...
                            * ones(length(idxwXsc{idxXsc}), 1, net.dtype);
                    elseif strcmp(net.initParamType, 'He')
                        Swx{idxXsc} = (1 / (fanIn)) * ...
                            ones(length(idxwXsc{idxXsc}), 1, net.dtype);
                    end
                    mwx{idxXsc} = randn(length(Swx{idxXsc}), 1) ...
                        .* sqrt(Swx{idxXsc});
                    if ~isempty(idxbXsc{idxXsc})
                        Sbx{idxXsc} = 1E-6 * ones(length(idxbXsc{idxXsc}), 1, net.dtype);
                        mbx{idxXsc} = 0 * randn(length(Sbx{idxXsc}), 1) ...
                            .* sqrt(Sbx{idxXsc});
                    end                   
                    if net.gpu
                        mwx{idxXsc} = gpuArray(mwx{idxXsc});
                        Swx{idxXsc} = gpuArray(Swx{idxXsc});
                        mbx{idxXsc} = gpuArray(mbx{idxXsc});
                        Sbx{idxXsc} = gpuArray(Sbx{idxXsc});
                    end
                end
                clear fanIn
                % Send to gpu
                if net.gpu
                    mw{j-1} = gpuArray(mw{j-1});
                    Sw{j-1} = gpuArray(Sw{j-1});
                    mb{j-1} = gpuArray(mb{j-1});
                    Sb{j-1} = gpuArray(Sb{j-1});                    
                end
            end 
            [mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx] = tagi.catParameters(mw, Sw,...
                mb, Sb, mwx, Swx, mbx, Sbx);
           theta = tagi.compressParameters(mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx); 
        end      
        function states = initializeStates(nodes, B, rB, xsc, dtype, gpu)
            % Normal net
            numLayers = length(nodes);          
            mz  = tagi.createStateCellarray(nodes, numLayers, B, rB, 0, dtype, gpu); 
            Sz  = mz; 
            ma  = mz;
            Sa  = mz;
            J   =  tagi.createStateCellarray(nodes, numLayers, B, rB, 1, dtype, gpu);
            % Residual net
            idx = xsc~=0;
            mdxs = cell(numLayers, 1);
            mdxs(idx) = mz(idx);
            Sdxs = mdxs;
            mxs  = mdxs;
            Sxs  = mdxs;
            states = tagi.compressStates(mz, Sz, ma, Sa, J, mdxs, Sdxs, mxs, Sxs);
        end
        function [deltaxs, deltadxs] = initializeShortcutStateDelta(xsc,...
                idxXsc, x, B, rB)
            layers   = xsc(xsc~=0);
            deltaxs  = cell(length(xsc), 1);
            deltadxs = cell(length(xsc), 1);
            for j = layers
                if ~isempty(idxXsc{j})
                    deltaxs{j}  = zeros(length(idxXsc{j})*B, rB, 'like', x{j});
                    deltadxs{j} = deltaxs{j};
                else
                    deltadxs{j} = zeros(size(x{j}), 'like', x{j});
                    deltaxs{j}  = zeros(size(x{j}), 'like', x{j});
                end
            end
        end
        function states = initializeInputs(states, mz0, Sz0, ma0, Sa0, J0,...
                mdxs0, Sdxs0, mxs0, Sxs0, xsc)
            [mz, Sz, ma, Sa, J, mdxs, Sdxs, mxs, Sxs] = tagi.extractStates(states);
            % Normal net
            mz{1} = mz0;
            if any(isempty(Sz0))
                Sz{1} = zeros(size(mz0), 'like', mz0);
            else
                Sz{1} = Sz0;
            end
            if any(isempty(ma0))
                ma{1} = mz0;
            else
                ma{1} = ma0;
            end 
            if any(isempty(Sa0))
                Sa{1} = Sz{1};
            else
                Sa{1} = Sa0;
            end   
            if any(isempty(J0))
                J{1} = ones(size(mz0), 'like', mz0);
            else
                J{1} = J0;
            end  
            % Residual net
            if any(isempty(mdxs0))&&~all(xsc==0)
                mdxs{1} = mz0;
            else
                mdxs{1} = mdxs0;
            end
            if any(isempty(Sdxs0))&&~all(xsc==0)
                Sdxs{1} = zeros(size(mz0), 'like', mz0);
            else
                Sdxs{1} = Sdxs0;
            end
            if any(isempty(mxs0))&&~all(xsc==0)
                mxs{1} = mz0;
            else
                mxs{1} = mxs0;
            end
            if any(isempty(Sxs0))&&~all(xsc==0)
                Sxs{1} = zeros(size(mz0), 'like', mz0);
            else
                Sxs{1} = Sxs0;
            end
            states = tagi.compressStates(mz, Sz, ma, Sa, J, mdxs, Sdxs, mxs, Sxs);
        end
        function maxIdx = initializeMaxPoolingIndices(nodes, layers,...
                layerEncoder, B, rB, dtype, gpu)
            if gpu
                zeroPad = zeros(1, 1, dtype, 'gpuArray');
            else
                zeroPad = zeros(1, 1, dtype);
            end
            numLayers = length(nodes);
            maxIdx = cell(numLayers, 1);
            maxPoolingLayers = find(layers==layerEncoder.mp);
            if ~isempty(maxPoolingLayers)
                for j = maxPoolingLayers
                    maxIdx{j} = zeros(nodes(j)*B, rB, 'like', zeroPad);
                end
            end
        end
        function normStat = initializeNormStat(nodes, filter, B, rB, layers,...
                layerEncoder, x)
            numLayers = length(nodes);
            mra = cell(numLayers, 1);
            layNorm = layers==layerEncoder.ln;
            batNormConv = layers==layerEncoder.bn&(layers==layerEncoder.conv|layers==layerEncoder.tconv|layers==layerEncoder.mp|layers==layerEncoder.ap);
            batNormfc = layers==layerEncoder.bn&layers==layerEncoder.fc;
            for j = layNorm
                mra{j} = zeros(B, rB, 'like', x);
            end
            for j = batNormfc
                mra{j} = zeros(nodes(j), rB, 'like', x);
            end
            for j = batNormConv
                mra{j} = zeros(filter(j), rB, 'like', x);
            end
            Sra = mra;
            normStat = tagi.compressNormStat(mra, Sra);
        end  
        function deltaTheta = initializeDeltaTheta(theta, rB, numLayers)
            deltaTheta = cell(numLayers-1, 1);
            for j = 1:numLayers-1
                deltaTheta{j} = repmat(theta{j}, [1, rB]);
            end
        end
        function [mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx] = catParameters(mw,...
                Sw, mb, Sb, mwx, Swx, mbx, Sbx)
            mw  = cat(1, mw{:});
            Sw  = cat(1, Sw{:});
            mb  = cat(1, mb{:});
            Sb  = cat(1, Sb{:});
            mwx = cat(1, mwx{:});
            Swx = cat(1, Swx{:});
            mbx = cat(1, mbx{:});
            Sbx = cat(1, Sbx{:});
        end
        function [mw, Sw, mb, Sb, mwx, Swx,...
                mbx, Sbx] = distributeParameters2Layers(mw, Sw, mb, Sb, mwx,...
                Swx, mbx, Sbx, numParams)
            mw  = mat2cell(mw, numParams(1, :));
            Sw  = mat2cell(Sw, numParams(1, :));
            mb  = mat2cell(mb, numParams(2, :));
            Sb  = mat2cell(Sb, numParams(2, :));
            mwx = mat2cell(mwx, numParams(3, :));
            Swx = mat2cell(Swx, numParams(3, :));
            mbx = mat2cell(mbx, numParams(4, :));
            Sbx = mat2cell(Sbx, numParams(4, :));
        end           
               
        % Storing
        function states = compressStates(mz, Sz, ma, Sa, J, mdxs, Sdxs, mxs,...
                Sxs)
            states = cell(9, 1);
            states{1} = mz;
            states{2} = Sz;
            states{3} = ma;
            states{4} = Sa;
            states{5} = J;
            states{6} = mdxs;
            states{7} = Sdxs;
            states{8} = mxs;
            states{9} = Sxs;
        end
        function [mz, Sz, ma, Sa, J, mdxs, Sdxs,...
                mxs, Sxs] = extractStates(states)
            mz   = states{1};
            Sz   = states{2};
            ma   = states{3};
            Sa   = states{4};
            J    = states{5};
            mdxs = states{6};
            Sdxs = states{7};
            mxs  = states{8};
            Sxs  = states{9};
        end
        function [mz, Sz, ma, Sa, J, mdxs, Sdxs,...
                mxs, Sxs] = extractStatesMultiGPUs(states)
            spmd
                mz   = states{1};
                Sz   = states{2};
                ma   = states{3};
                Sa   = states{4};
                J    = states{5};
                mdxs = states{6};
                Sdxs = states{7};
                mxs  = states{8};
                Sxs  = states{9};
            end
        end
        function theta = compressParameters(mw, Sw, mb, Sb, mwx, Swx, mbx,...
                Sbx)
            theta     = cell(8, 1);
            theta{1}  = mw;
            theta{2}  = Sw;
            theta{3}  = mb;
            theta{4}  = Sb;
            theta{5}  = mwx;
            theta{6}  = Swx;
            theta{7}  = mbx;
            theta{8}  = Sbx;
        end
        function [mw, Sw, mb, Sb, mwx, Swx,...
                mbx, Sbx] = extractParameters(theta)
            mw  = theta{1};
            Sw  = theta{2};
            mb  = theta{3};
            Sb  = theta{4};
            mwx = theta{5};
            Swx = theta{6};
            mbx = theta{7};
            Sbx = theta{8};
        end
        function [mw, Sw, mb, Sb, mwx, Swx,...
                mbx, Sbx] = extractParameters_V2(theta)
            mw  = theta(1, :);
            Sw  = theta(2, :);
            mb  = theta(3, :);
            Sb  = theta(4, :);
            mwx = theta(5, :);
            Swx = theta(6, :);
            mbx = theta(7, :);
            Sbx = theta(8, :);
        end
        function normStat = compressNormStat(mra, Sra)
            normStat = cell(2, 1);
            normStat{1} = mra;
            normStat{2} = Sra;
        end
        function [mra, Sra] = extractNormStat(normStat)
            mra = normStat{1};
            Sra = normStat{2};
        end   
        
        % Create cell with an array
        function x = createInitCellwithArray(numLayers, dtype, gpu)
            x = cell(numLayers, 1);
            if gpu
                x(:) = {gpuArray(nan(1, 1, dtype))};
            else
                x(:) = {nan(1, 1, dtype)};
            end
        end
        function z = createStateCellarray(nodes, numLayers, B, rB, value,...
                dtype, gpu)   
            z = cell(numLayers, 1);
            if gpu
                zeroPad = zeros(1,1,dtype, 'gpuArray');
            else
                zeroPad = zeros(1,1,dtype);
            end
            for j = 2:numLayers               
                z{j} = zeros(nodes(j)*B, rB, 'like', zeroPad) + value;
            end
        end 
        function d = createDevCellarray(nodes, numLayers, B, rB, dtype, gpu)   
            d = cell(numLayers, 1);
            if gpu
                onePad = ones(1,1,dtype, 'gpuArray');
            else
                onePad = ones(1,1,dtype);
            end
            for j = 1:numLayers               
                d{j} = ones(nodes(j)*B, rB, 'like', onePad);
            end
%             d{numLayers} = zeroPad + 1;
        end
        function normStat = createInitNormStat(net)
            mra   = cell(length(net.nodes) -1, 1);            
            Sra   = cell(length(net.nodes) -1, 1);
            dtype = net.dtype;
            numLayers = length(net.layer);
            layer     = net.layer;
            if net.gpu
                zeroPad = zeros(1,1,dtype, 'gpuArray');
            else
                zeroPad = zeros(1,1,dtype);
            end
            for j = 2:numLayers 
                if layer(j-1) == net.layerEncoder.conv
                    if layer(j) == net.layerEncoder.ln
                        mra{j-1} = zeros(net.batchSize * net.repBatchSize, 1, 'like', zeroPad);
                        Sra{j-1} = ones(net.batchSize * net.repBatchSize, 1, 'like', zeroPad);
                    elseif layer(j) == net.layerEncoder.bn
                        mra{j-1} = zeros(net.filter(j), 1, 'like', zeroPad);
                        Sra{j-1} = ones(net.filter(j), 1, 'like', zeroPad);
                    else
                        mra{j-1} = zeros(1, 1, 'like', zeroPad);
                        Sra{j-1} = ones(1, 1, 'like', zeroPad);
                    end
                elseif layer(j-1) == net.layerEncoder.fc
                    if layer(j) == net.layerEncoder.ln
                        mra{j-1} = zeros(net.batchSize * net.repBatchSize, 1, 'like', zeroPad);
                        Sra{j-1} = ones(net.batchSize * net.repBatchSize, 1, 'like', zeroPad);
                    elseif layer(j) == net.layerEncoder.bn
                        mra{j-1} = zeros(net.nodes(j), 1, 'like', zeroPad);
                        Sra{j-1} = ones(net.nodes(j), 1, 'like', zeroPad);
                    else
                        mra{j-1} = zeros(1, 1, 'like', zeroPad);
                        Sra{j-1} = ones(1, 1, 'like', zeroPad);
                    end
                end
            end
            normStat = tagi.compressNormStat(mra, Sra);
        end
    end
end