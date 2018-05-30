%% DTI classification demo ISMRM 2018
% 
% Created in May 2018 for ISMRM educational "How to Jump-Start Your Deep Learning Research"
% Educational course Deep Learning: Everything You Want to Know, Saturday, June 16th 2018
% Joint Annual meeting of ISMRM and ESMRMB, Paris, France, June 16th to 21st
% 
% Created with Matlab 2018a
% 
% florian.knoll@nyumc.org
% 

clear all; close all; clc;
rng default;

%% Load data
% The first case is used as an independent test set. Cases 2-4 are used for training and validation
%
% Entries in the CVS file are
% 1: sample
% 2: row
% 3: column
% 4: slice
% 5: T1 weighted anatomical image
% 6: FA
% 7: MD
% 8: AD
% 9: RD
% 10: Label
% 
% Classes are
% 1: left thalamus
% 1: left genu of the corpus callosum
% 1: left subcortical white matter of inferior frontal gyrus
data1 = csvread('./data/dti/sampledata100206.csv');
data2 = csvread('./data/dti/sampledata105620.csv');
data3 = csvread('./data/dti/sampledata107725.csv');
data4 = csvread('./data/dti/sampledata112314.csv');
data_cat = [data2; data3; data4];

%% Data reorganization and subset selection
% Remove classes and slice position features
X = data_cat(:,5:(end-1))';
Y = data_cat(:,end)';
x_test = data1(:,5:(end-1))';
y_test = data1(:,end)';
clear data1; clear data2; clear data3; clear data4; clear data_cat;

nFeatures = size(x_test,1);
nClasses = max(y_test);

[~,nSamples] = size(X);

%% Normalize data
for ii=1:nFeatures
    feature_normalization = max(X(ii,:));
    X(ii,:) = X(ii,:)/feature_normalization;
    x_test(ii,:) = x_test(ii,:)/feature_normalization;
end

%% Separate validation data set and shuffle
index = randperm(nSamples);
setsize = ceil(nSamples*0.8);
indextrain = index(1:setsize);
indexval = index((setsize+1):length(index));

x_train = X(:,indextrain);
x_val = X(:,indexval);

y_train = Y(:,indextrain);
y_val = Y(:,indexval);

clear X; clear Y;

%% Prepare matrices for training function
y_train = categorical(y_train)';
y_val = categorical(y_val)';
y_test = categorical(y_test)';

x_train = permute(x_train,[1,3,4,2]);
x_val = permute(x_val,[1,3,4,2]);
x_test = permute(x_test,[1,3,4,2]);

%% NN with 3 HIDDEN LAYER NEURONS
nElements = 100;
nLayers = 3;
inputLayer=imageInputLayer([nFeatures,1,1]);
f1=fullyConnectedLayer(nElements);
f2=fullyConnectedLayer(nElements);
f3=fullyConnectedLayer(nElements);
f4=fullyConnectedLayer(nClasses);
s1=softmaxLayer();
outputLayer=classificationLayer();

architecture = [inputLayer; f1; f2; f3; f4; s1; outputLayer];
disp(architecture);

epochs = 250;
miniBatchSize = 1024;
InitialLearnRate = 0.001;

% Training options: Note that we set the validation patience stopping
% criterion to the number of epochs. This is a stupid thing to do, but we
% want force the training to go to the defined number of epochs so that it
% is consistent with Tensorflow and Pytorch
options = trainingOptions('adam','MaxEpochs',epochs,'InitialLearnRate',InitialLearnRate,...
    'MiniBatchSize',miniBatchSize,'ExecutionEnvironment','cpu','Plots','training-progress',...
    'ValidationData',{x_val,y_val},'ValidationPatience',epochs);

%% Train
tic
[net,op] = trainNetwork(x_train,y_train,architecture,options);
toc 

%% Evaluate
output_train = classify(net, x_train);
output_val = classify(net, x_val);
output_test = classify(net, x_test);

% Calculate Accuracy
acc_train = sum(double(output_train) == double(y_train))/length(double(y_train));
acc_val = sum(double(output_val) == double(y_val))/length(double(y_val));
acc_test = sum(double(output_test) == double(y_test))/length(double(y_test));
disp(['acc train: ', num2str(acc_train)]);
disp(['acc val: ', num2str(acc_val)]);
disp(['acc test: ', num2str(acc_test)]);

%% Export figure
figure_handle=findall(groot, 'Type', 'Figure');
saveas(figure_handle(2),['./training_plots_matlab/dti_FC_',num2str(nLayers),'layers_',num2str(nElements),'elements_epochs',num2str(epochs),'.png']);

