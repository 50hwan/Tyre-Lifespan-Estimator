%% 0. PREPROCESSING
imds = imageDatastore('dataset', ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.7, 0.3, 'randomized');

net_vgg19 = vgg19;
inputSize_vgg19 = net_vgg19.Layers(1).InputSize;

layersTransfer_vgg19 = net_vgg19.Layers(1:end-3);

numClasses = numel(categories(imdsTrain.Labels));

layers_vgg19 = [
    layersTransfer_vgg19
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);

augimdsTrain_vgg19 = augmentedImageDatastore( ...
    inputSize_vgg19(1:2),imdsTrain, ... 
    'DataAugmentation',imageAugmenter, ...
    'ColorPreprocessing', 'gray2rgb');

augimdsValidation_vgg19 = augmentedImageDatastore( ...
    inputSize_vgg19(1:2),imdsValidation, ...
    'ColorPreprocessing', 'gray2rgb');

%% 1. TRAIN MAIN

options_vgg19 = trainingOptions('sgdm', ...
    'MiniBatchSize',30, ...
    'MaxEpochs',10, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation_vgg19, ...
    'ValidationFrequency',3, ...
    'InitialLearnRate', 2e-4, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 5, ...
    'Verbose',false, ...
    'Plots','training-progress');

netTransfer_vgg19 = trainNetwork(augimdsTrain_vgg19,layers_vgg19,options_vgg19);


%% 2. TEST MAIN (imdsTest 존재시)

augimdsTest_vgg19 = augmentedImageDatastore( ...
    inputSize_vgg19(1:2),imdsTest, ...
    'ColorPreprocessing', 'gray2rgb');

YPred_vgg19 = classify(netTransfer_vgg19, augimdsTest_vgg19);
YTest_vgg19 = imdsTest.Labels;

accuracy_vgg19 = mean(YPred_vgg19 == YTest_vgg19);
fprintf('test data acc of vgg-19: %.2f%%\n', accuracy_vgg19 * 100);


% 이미지를 네트워크에 전달하여 확률 예측
YProb_vgg19 = predict(netTransfer_vgg19, augimdsTest_vgg19);
classNames_vgg19 = netTransfer_vgg19.Layers(end).ClassNames;

% datas
data_vgg19 = [string(cellstr(YPred_vgg19)), num2cell(YProb_vgg19), num2cell(YPred_vgg19 == YTest_vgg19)];


%% 3. TEST MEASURE (직접 만든 test data - linear regression 필요)

% 파라미터 정의
folder_name = 'test_main';  % 이미지가 있는 폴더 이름
scores = [0, 1];  % 각 클래스에 대한 점수

% 폴더 내의 모든 이미지 파일을 처리
imageFiles = dir(fullfile(folder_name, '*.jpg'));  % 이미지 파일 형식에 따라 수정

% 클래스 이름
classNames = netTransfer_vgg19.Layers(end).ClassNames;

% 각 이미지에 대해 예측 및 점수 계산
ai_scores = zeros(1, length(imageFiles));  % AI 모델을 통해 얻은 점수를 저장할 배열
for i = 1:length(imageFiles)
    % 이미지 로딩 및 크기 조정
    testImage = imread(fullfile(folder_name, imageFiles(i).name));
    testImage = imresize(testImage, [224 224]);

    % 이미지를 네트워크에 전달하여 확률 예측
    YPred = predict(netTransfer_vgg19, testImage);
    
    % 점수 계산
    ai_scores(i) = sum(YPred .* scores);  % AI 모델을 통해 얻은 점수를 저장
end



% 마모한계선까지 거리, ground truth
tire_grooves = [6.80, 9.61, 7.25, 9.23, 8.70, 9.14, 8.63, 8.47, 8.05, 4.03, 4.76, 6.46, 3.90, 5.78, 5.28];
% 영점조절
zero_point = 2; 
% 2번째부터 9번째까지 영점 조정 완료
tire_grooves_cal = tire_grooves;
tire_grooves_cal(2:9) = tire_grooves(2:9) - zero_point;


% 데이터 길이 일치 확인
if length(tire_grooves_cal) ~= length(ai_scores)
    error('두 데이터의 길이가 일치하지 않습니다.');
end

% 데이터 플로팅
figure;
plot(tire_grooves_cal, ai_scores, 'o');
xlabel('Tire Grooves');
ylabel('AI Scores');
title('Tire Grooves vs AI Scores');

% 선형 피팅 및 R 값 계산
fit_result = polyfit(tire_grooves_cal, ai_scores, 1);
[R, p_value] = corrcoef(tire_grooves_cal, ai_scores);
disp(['R value: ', num2str(R(1,2))]);
disp(['p value: ', num2str(p_value(1,2))]);

%% 4. TEST MEASURE V2 (divide & process)

% 파라미터 정의
folder_name = 'test_cut_36'; % 이미지가 있는 폴더 이름
scores = [0, 1]; % 각 클래스에 대한 점수

% 폴더 내의 모든 서브 폴더를 처리
subFolders = dir(folder_name); 
subFolders = subFolders([subFolders.isdir]);
subFolders = subFolders(~ismember({subFolders.name},{'.','..'})); % '.' 과 '..' 제거

% 클래스 이름
classNames = netTransfer_vgg19.Layers(end).ClassNames;

% 각 서브 폴더에 대해 예측 및 점수 계산
ai_scores = zeros(1, length(subFolders)); % AI 모델을 통해 얻은 점수를 저장할 배열

for i = 1:length(subFolders)

    % 서브 폴더 내의 모든 이미지 파일을 처리
    imageFiles = dir(fullfile(folder_name, subFolders(i).name, '*.jpg')); % 이미지 파일 형식에 따라 수정
    
    sub_scores = zeros(1, length(imageFiles)); % 각 이미지에 대한 점수를 저장할 배열

    for j = 1:length(imageFiles)

        % 이미지 로딩 및 크기 조정
        testImage = imread(fullfile(folder_name, subFolders(i).name, imageFiles(j).name));
        testImage = imresize(testImage, [224 224]);

        % 이미지를 네트워크에 전달하여 확률 예측
        YPred = predict(netTransfer_vgg19, testImage);

        % 점수 계산
        sub_scores(j) = sum(YPred .* scores); % AI 모델을 통해 얻은 점수를 저장
    end

    % 폴더별 점수 평균 계산
    ai_scores(i) = mean(sub_scores);
end


% 현재 (조금 갈린) 타이어의 골짜기?(트레드 패턴 이라 한다네) 길이 (직접 잰거)
tire_grooves = [3.30, 7.11, 4.75, 6.73, 6.20, 5.14, 6.13, 5.97, 5.55, 4.03, 4.76, 3.46, 3.90, 5.78, 5.28];

% 새삥 타이어 트레드 패턴 길이 (8mm, 한국타이어 승용차용이라는데 나중에 알아보셈, 틀리면 다른 값 집어넣으십쇼)
new_tire_groove = 9;
% 헌 타이어 마모한계선 (1.6mm)
old_tire_groove = 1.6;

% 마모한계선이 1.6이니까, 타이어가 마모되면서 트레드 패턴 깊이가 8에서 1.6까지 줄어듬
% 그니까 각 타이어 트레드 패턴 길이 잰 걸로 수명을 파악하면
tire_lifespan = (tire_grooves - old_tire_groove) / (new_tire_groove - old_tire_groove);




% 데이터 길이 일치 확인
if length(tire_lifespan) ~= length(ai_scores)
    error('두 데이터의 길이가 일치하지 않습니다.');
end

% 데이터 플로팅
figure;
plot(tire_lifespan, ai_scores, 'o');
xlabel('Tire Remaining Lifespan');
ylabel('AI Scores');
title('Tire Remaining Lifespan vs AI Scores');

% 선형 피팅 및 R 값 계산
fit_result = polyfit(tire_lifespan, ai_scores, 1);
[R, p_value] = corrcoef(tire_lifespan, ai_scores);
disp(['R value: ', num2str(R(1,2))]);
disp(['p value: ', num2str(p_value(1,2))]);


