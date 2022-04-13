%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         dp
% Description:  data processing
% Authors:      James-A. Goulet & Luong-Ha Nguyen
% Created:      November 8, 2019
% Updated:      August 27, 2021
% Contact:      james.goulet@polymtl.ca & luongha.nguyen@gmail.com 
% Copyright (c) 2021 Luong-Ha Nguyen & James-A. Goulet. All rights reserved. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
classdef dp
    methods (Static)             
        % Data loader
        function [img, y, labels, udIdx] = imagenetDataLoader(idx, imdb,...
                da, imgSize, resizedSize, nx, ny, nye, B, rB, dtype,...
                trainMode, resizeMode, gpu, cuda)           
            idx = double(idx);           
            run = true;
            % Get batch of data
            if strcmp(imdb.imgExt, 'mat')
                img = imdb.img(:, :, :, idx);
            else
                img = zeros([double(imgSize), rB*B], 'uint8');
                if resizeMode
                    win = centerCropWindow2d(double(resizedSize(1:2)),...
                        double(imgSize(1:2)));
                    for i = 1:rB * B
                        while run
                            try
                                imgloop = readimage(imdb.img, idx(i));
                                checked = true;
                            catch
                                idx(i) = randperm(imdb.numImages, 1);
                                checked = false;
                            end
                            if checked; break; end
                        end
                        imgloop = imresize(imgloop, resizedSize(1:2));
                        if length(size(imgloop))==length(size(resizedSize))
                            imgloop = repmat(imgloop, [1 1 resizedSize(3)]);
                        end
                        imgloop = imcrop(imgloop, win);
                        img(:,:,:,i) = imgloop;
                    end
                else
                    for i = 1 : rB * B
                        while run
                            try
                                imgloop = readimage(imdb.img, idx(i));
                                checked = true;
                            catch
                                idx(i) = randperm(imdb.numImages, 1);
                                checked = false;
                            end
                            if checked; break; end
                        end
                        if length(size(imgloop))==length(size(imgSize))
                            imgloop = repmat(imgloop, [1 1 resizedSize(3)]);
                        end
                        img(:,:,:,i) = imgloop;
                    end
                end
            end
            
            % Data augmentation
            if da.enable && trainMode
                for i = 1:rB*B
                    img(:,:,:,i) = dp.dataAugmentation(img(:,:,:,i), da);
                end
            end
            
            % Labels and encoding observations
            labels = single(imdb.digitlabels(idx));
            if trainMode
                y     = imdb.classObs(labels+1, :);
                enIdx = imdb.classIdx(labels+1, :);
                if cuda
                    udIdx =  enIdx';                   
                else
                    udIdx = dp.selectIndices(enIdx, B*rB, ny, dtype);
                end
                y     = reshape(y', [nye*B, rB]);
                udIdx = reshape(udIdx, [numel(udIdx)/rB, rB]);
            else
                y     = nan;
                udIdx = nan;
            end
            udIdx = cast(udIdx, dtype);
            y     = cast(y, dtype);
            
            % Normalize data
            img = cast(img, dtype);
            img = img./255;
            img = (img - imdb.imgStat(:,:,:,1))./imdb.imgStat(:,:,:,2);           
            
            % Flatten
            img = reshape(img, [nx*B, rB]);
            
            % Transfer to GPUs
            if gpu
                img = gpuArray(img);
                y   = gpuArray(y); 
                udIdx = gpuArray(udIdx);
            end                       
        end       
        function x = dataLoader(x, da, B, rB, trainMode)
            if da.enable&&trainMode
                for n = 1:B:rB*B
                    idx = n:n+B-1;
                    x(:,:,:,idx) = dp.dataAugmentation(x(:,:,:, idx), da);
                end
            end
            x = reshape(x, [numel(x), 1]);
        end
        function labels = convertLabels(labeldb, reflabels, numClasses, dtype)           
            labels = zeros(numel(labeldb), 1, dtype);
            for c = 1:numClasses
               labels(labeldb==reflabels(c)) = c-1;
            end
        end       
        function [xcell, labelcell, labelStat] = regroupClasses(x, labels)
            minC = min(labels);
            maxC = max(labels);
            numClasses = maxC - minC + 1;
            xcell     = cell(numClasses, 1);
            labelcell = cell(numClasses, 1);
            labelStat = zeros(numClasses, 1);
            for c = 1:numClasses
                idx = labels==c-1;
                labelStat(c) = sum(idx);
                xcell{c}     = x(:,:,:,idx);
                labelcell{c} = labels(idx);
            end
        end
        function [xs, ls] = selectSamples(x, labels, numSamples)            
            [xcell, labelcell, labelStat] = dp.regroupClasses(x, labels);
            numClasses = size(xcell, 1);
            xs = cell(numClasses, 1);
            ls = cell(numClasses, 1);
            for c = 1:numClasses
               idx = randperm(labelStat(c), numSamples); 
               xs{c} = xcell{c}(:,:,:,idx);
               ls{c} = labelcell{c}(idx);
            end
            xs = cat(4, xs{:});
            ls = cat(1, ls{:});
        end
        
        % Data Augmentation
        function x = dataAugmentation(x, da)
            for k = 1:length(da.types)
                if da.types(k) == da.horizontalFlip
                    x = dp.imgFlip(x, 2, da.p(k));
                end
                if da.types(k) == da.randomCrop
                    x = dp.randomCrop(x, da.randomCropPad, da.randomCropSize, da.p(k));
                end
                if da.types(k) == da.verticalFlip
                    x = dp.imgFlip(x, 1, da.p(k));
                end
                if da.types(k) == da.cutout
                    x = dp.cutout(x, da.cutoutSize, da.p(k));
                end
            end
        end
        function x = imgFlip(x, dim, p)
            a = binornd(1, p); 
            if dim == 1 %&& a == 1
                tform2 = randomAffine2d('YReflection',true);
                x = imwarp(x,tform2);
%                 x = x(end:-s1:1,:, :, :);
            elseif dim == 2 %&& a == 1
                tform2 = randomAffine2d('XReflection',true);
                x = imwarp(x,tform2);
%                 x = x(:,end:-1:1,:, :);
            end
        end
        function x = randomCrop(x, pad, cropSize, p)
            wout = cropSize(1);
            hout = cropSize(2);
            win  = wout+2*pad;
            hin  = hout+2*pad;
            a    = binornd(1, p); 
            [w, h, d, n] = size(x);
            if a == 1
                paddingImg = zeros(w+2*pad, h+2*pad, d, n, class(x));
                paddingImg(pad+1:end-pad, pad+1:end-pad, :, :) = x;
                wd = randomCropWindow2d([win, hin], [wout hout]);
                x = imcrop(paddingImg, wd);
%                 idxW = randi(w+2*pad-wout+1)+(0:wout-1);
%                 idxH = randi(h+2*pad-hout+1)+(0:hout-1);
%                 x    = paddingImg(idxW, idxH, :, :);
            end
        end
        function x = cutout(x, rect, p)           
            a = binornd(1, p); 
            [w, h, d, n] = size(x); 
            if a == 1
                wcut = rect(1);
                hcut = rect(2);
                idxW = randi(w-wcut+1)+(0:wcut-1);
                idxH = randi(h-hcut+1)+(0:hcut-1);
                x(idxW, idxH, :, :) = zeros(wcut, hcut, d, n, 'like', x);
            end
        end
        function x = cvr2img(x, imgSize)
            w = imgSize(1);
            h = imgSize(2);
            d = imgSize(3);
            n = size(x, 1); 
            x = reshape(reshape(x',[w*h*d*n, 1]), [w, h, d, n]);
        end
        
        % Normalization
        function x = normalizeRGB(x, m, s, padding)
            D    = size(x, 3);
            for i = 1:D
                x(:,:,i,:)= (x(:,:,i,:) -  m(i))./s(i);
            end
            if padding>0
                x = dp.addPadding(x, padding);
            end
        end
        function x = denormalizeImg(x, m, s)
            D    = size(x, 3);
            for i = 1:D
                x(:,:,i,:)= x(:,:,i,:).*s(i) + m(i);
            end
        end
        function [xntrain, yntrain, xntest, yntest, mxtrain, sxtrain,...
                mytrain, sytrain] = normalize(xtrain, ytrain, xtest, ytest)
            mxtrain = nanmean(xtrain);
            sxtrain = nanstd(xtrain);
            idx     = sxtrain==0;
            sxtrain(idx) = 1;
            mytrain = nanmean(ytrain);
            sytrain = nanstd(ytrain);
            xntrain = (xtrain - mxtrain)./sxtrain;
            yntrain = (ytrain - mytrain)./sytrain;
            xntest  = (xtest - mxtrain)./sxtrain;
            yntest  = ytest;
        end
        function [y, sy] = denormalize(yn, syn, myntrain, syntrain)
            y   = yn.*syntrain + myntrain;
            if ~isempty(syn)
                sy  = (syntrain.^2).*syn;
            else
                sy  = [];
            end
        end
        
        % Data preparation
        function [imdb] = prepareTrainingDatabase(pathFolder, numClasses,...
                imgExt, dtype)
            img = imageDatastore(pathFolder, ...
                'LabelSource','foldernames',...
                'IncludeSubfolders',true, 'FileExtensions', {imgExt});
            [classObs, classIdx] = dp.class_encoding(numClasses);            
            T           = countEachLabel(img);
            reflabels   = categorical(T{:, 1});
            digitlabels = dp.convertLabels(img.Labels, reflabels, numClasses, dtype); 
            
            % Add to database
            imdb.img          = img;
            imdb.reflabels    = reflabels;
            imdb.digitlabels  = digitlabels;
            imdb.classObs     = classObs;
            imdb.classIdx     = classIdx;
            imdb.numOutputs   = max(max(classIdx));
            imdb.numEnOutputs = size(classObs, 2);
            imdb.numClasses   = numClasses;
            imdb.numImages    = length(img.Labels);
            imdb.imgExt       = imgExt;
        end
        function imdb = prepareValDatabase(pathFolder, vallabels, reflabels,...
                numClasses, classObs, classIdx, imgExt, dtype)
            img = imageDatastore(pathFolder, ...
                'LabelSource','foldernames',...
                'IncludeSubfolders',true, 'FileExtensions', {imgExt});  
            if ~isempty(reflabels)
                digitlabels = dp.convertLabels(vallabels, reflabels, numClasses, dtype); 
            else
                digitlabels = vallabels;
            end
            % Add to database
            imdb.img         = img;
            imdb.digitlabels = digitlabels;
            imdb.numImages   = length(img.Files);
            imdb.classObs    = classObs;
            imdb.classIdx    = classIdx;
            imdb.imgExt      = imgExt;
        end
        function imdb = prepareTestDatabase(pathFolder, classObs, classIdx,...
                imgExt)
            img = imageDatastore(pathFolder, ...
                'LabelSource','foldernames',...
                'IncludeSubfolders',true, 'FileExtensions', {imgExt}); 
            imdb.img       = img;
            imdb.numImages = length(img.Files);
            imdb.classObs  = classObs;
            imdb.classIdx  = classIdx;
            imdb.imgExt    = imgExt;
        end
        function imdb = prepareDatabass(img, imgSize, labels, numClasses,...
                imgExt) 
            img = dp.cvr2img(img, imgSize);
%             img = permute(img, [2 1 3 4]);
            [classObs, classIdx] = dp.class_encoding(numClasses);  
            
            % Add to database
            imdb.img          = img;
            imdb.reflabels    = [];
            imdb.digitlabels  = labels;
            imdb.classObs     = classObs;
            imdb.classIdx     = classIdx;
            imdb.numOutputs   = max(max(classIdx));
            imdb.numEnOutputs = size(classObs, 2);
            imdb.numClasses   = numClasses;
            imdb.numImages    = size(img, 4);
            imdb.imgExt       = imgExt;
        end
        
        % Shared functions
        function [m, s] = meanstd(x)           
            nObs = size(x, 4);
            D    = size(x, 3);
            H    = size(x, 2);
            W    = size(x, 1);
            x    = permute(x, [1 2 4 3]);
            x    = reshape(x, [nObs*H*W, D]);
            m    = mean(x);
            s    = std(x);
        end
        function xp = addPadding(x, padding)
            nObs = size(x, 4);
            D    = size(x, 3);
            H    = size(x, 2);
            W    = size(x, 1);
            xp   = zeros(H+padding, W+padding, D, nObs);
            for k = 1:nObs
                for i = 1:D
                    xp(1:W,1:H,i,k)= x(:,:,i,k);
                end
            end
        end             
        function [xtrain, ytrain, xtest, ytest] = split(x, y, ratio)
            numObs      = size(x, 1);
            idxobs      = 1:numObs;
%             idxobs      = randperm(numObs);
            idxTrainEnd = round(ratio*numObs);
            idxTrain    = idxobs(1:idxTrainEnd);
            idxTest     = idxobs((idxTrainEnd+1):numObs);
            xtrain      = x(idxTrain, :);
            ytrain      = y(idxTrain, :);
            xtest       = x(idxTest, :);
            ytest       = y(idxTest, :);
        end
        function [trainIdx, testIdx] = indexSplit(numObs, ratio, dtype)
           idx = randperm(numObs);
           trainIdxEnd = round(numObs*ratio);
           trainIdx = idx(1:trainIdxEnd)';
           testIdx  = idx(trainIdxEnd+1:end)';
           if strcmp(dtype, 'single')
               trainIdx = int32(trainIdx);
               testIdx = int32(testIdx);
           end
        end
        function [x, y, labels, encoderIdx] = selectData(x, y, labels,...
                encoderIdx, idx)
            x = x(:,:,:,idx);
            if ~isempty(y)
                y = y(idx, :);
            else
                y = [];
            end
            if ~isempty(labels)
                labels = labels(idx, :);
            else
                labels = [];
            end
            if ~isempty(encoderIdx)
                encoderIdx = encoderIdx(idx, :);
            else
                encoderIdx = [];
            end
        end
        function foldIdx = kfolds(numObs, numFolds)
            numObsPerFold = round(numObs/(numFolds));
            idx           = 1:numObsPerFold:numObs;
            if ~ismember(numObs, idx)
                idx = [idx, numObs];
            end
            if length(idx)>numFolds+1
                idx(end-1) = []; 
            end
            foldIdx = cell(numFolds, 1);
            for i = 1:numFolds
                if i == numFolds
                    foldIdx{i} = [idx(i):idx(i+1)]';
                else
                    foldIdx{i} = [idx(i):idx(i+1)-1]';
                end
            end
        end       
        function [xtrain, xval] = regroup(x, foldIdx, valfold)
            trainfold       = 1:size(foldIdx, 1);
            trainfold(valfold) = [];
            xval            = x(foldIdx{valfold}, :);
            trainIdx        = cell2mat(foldIdx(trainfold));
            xtrain          = x(trainIdx, :);
        end      
        function y  = transformObs(y)
            maxy    = 10;
            miny    = -10;
            idx     = logical(y);
            y(idx)  = maxy;
            y(~idx) = miny;
        end
        function prob  = probFromloglik(loglik)
            maxlogpdf = max(loglik);
            w_1       = bsxfun(@minus,loglik,maxlogpdf);
            w_2       = log(sum(exp(w_1)));
            w_3       = bsxfun(@minus,w_1,w_2);
            prob      = exp(w_3);
        end
        
        % Classification encoder
        function [y, idx]   = encoder(yraw, numClasses, dtype)
            [~, idx_c]=dp.class_encoding(numClasses);
            y   = zeros(size(yraw, 1), max(max(idx_c)), dtype);
            if strcmp(dtype, 'single')
                idx = zeros(size(yraw, 1), size(idx_c, 2), 'int32');
            elseif strcmp(dtype, 'double')
                idx = zeros(size(yraw, 1), size(idx_c, 2), 'int64');
            end            
            for c = 1:numClasses
                idxClasses         = yraw==c-1;
                [idxLoop, obs]     = dp.class2obs(c, dtype, numClasses);
                y(idxClasses, idxLoop) = repmat(obs, [sum(idxClasses), 1]);
                idx(idxClasses, :) = repmat(idxLoop, [sum(idxClasses), 1]);
            end
        end
        function idx        = selectIndices(idx, batchSize, numClasses,...
                dtype)
            if strcmp(dtype, 'single')
                numClasses = single(numClasses);
                idx        = single(idx);
                batchSize  = single(batchSize);
            elseif strcmp(dtype, 'double')
                numClasses = double(numClasses);
                idx        = double(idx);
                batchSize  = single(batchSize);
            end
            for b = 1 : batchSize
                idx(b, :) = idx(b, :) + (b-1)*numClasses;
            end
            idx = reshape(idx', [size(idx, 1)*size(idx, 2), 1]);
        end
        function [obs, idx] = class_encoding(numClasses)
            H=ceil(log2(numClasses));
            C=fliplr(de2bi([0:numClasses-1],H));
            obs=(-1).^C;
            idx=ones(numClasses,H);
            C_sum=[zeros(1,H),numClasses];
            for h=H:-1:1
                C_sum(h)=ceil(C_sum(h+1)/2);
            end
            C_sum=cumsum(C_sum)+1;
            for i=1:numClasses
                for h=1:H-1
                    idx(i,h+1)=bi2de(fliplr(C(i,1:h)))+C_sum(h);
                end
            end
            max_idx=max(max(idx));
            unused_idx=setdiff(1:max_idx,idx);
            for j=1:length(unused_idx)
                idx_loop=(idx-j+1)>(unused_idx(j));
                idx(idx_loop)=idx(idx_loop)-1;
            end
        end
        function [idx, obs] = class2obs(class, dtype, numClasses)
            [obs_c, idx_c]=dp.class_encoding(numClasses);
            idx=idx_c(class,:);
            obs=obs_c(class,:);
            if strcmp(dtype, 'single')
                idx = int32(idx);
                obs = single(obs);
            elseif strcmp(dtype, 'half')
                idx = int32(idx);
                obs = half(obs);
            end
        end
        function p_class    = obs2class(mz, Sz, obs_c, idx_c)          
            alpha = 3;
            p_obs = normcdf(mz./sqrt((1/alpha)^2 + Sz), 0, 1);           
            p_class=prod(abs(p_obs(idx_c)-(obs_c==-1)), 2);
        end 
        
        % Visual field
        function [ytrain, ytest, labelTest, trainIdx,...
                testIdx] = prepareData4visualField(xtrain, xtest, yrefTrain,...
                yrefTest, ntrain, ntest, classes)
            nx     = size(xtrain, 2);
            ytrain = zeros(ntrain, nx, length(classes), 'like', xtrain);
            ytest  = zeros(ntest, nx, length(classes), 'like', xtrain);
            labelTest = zeros(ntest, length(classes), 'like', xtrain);
            trainIdx = [];
            testIdx  = [];
            for c = 1 : length(classes)
                % Training set
                idx         = find(yrefTrain == classes(c));
                idx         = idx(1:ntrain);
                ytrain(:, :, c)  = xtrain(idx, :);
                trainIdx    = [trainIdx, idx];
                
                % Test set
                idx            = find(yrefTest == classes(c));
                idx            = idx(1:ntest);
                ytest(:, :, c) = xtest(idx, :);
                labelTest(:, c)= yrefTest(idx, :); 
                testIdx    = [testIdx, idx];
            end
            ytest     = reshape(permute(ytest, [2 1 3]), [nx, ntest * length(classes)])';
            labelTest = reshape(labelTest, [ntest * length(classes) , 1]);
            idxperm   = randperm(ntest * length(classes), ntest * length(classes));
            ytest     = ytest(idxperm, :);
            labelTest = labelTest(idxperm);
            for c = 1 : length(classes)
                labelTest(labelTest == classes(c)) = c-1;
            end
        end
        
        % Verification
        function [ok, prob] = classCheck(classObs, classIdx, numOutputs, numClasses)
            labels = colon(1, numClasses);
            ok     = zeros(numClasses, 1, 'logical');
            prob   = zeros(numClasses, 1);
            Sl     = zeros(numOutputs, 1)+1E-6;
            for c = 1:numClasses
                ml = zeros(numOutputs, 1)+normrnd(0, 1, [202 1]);
                ml(classIdx(c, :)) = classObs(c, :);
                P = dp.obs2class(ml, Sl, classObs, classIdx);
                [prob(c), pc] = max(P);
                ok(c) = pc==labels(c);                
            end
        end
    end
end