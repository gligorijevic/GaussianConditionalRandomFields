%%load data

close all;
clc

rng(17); % set random seed 15 je ok
numTimeSteps = 2;
T = [1 numTimeSteps-1];
trainingTS = 1:1;
testcol = numTimeSteps;
N = 400; % have to be square of some number for grid and web structures
no_features = 10;
no_hiddenNeurons = 20;
alpha = 500; % bilo 100
beta = 1000;
sparseness = 1;
minSimilarityValue = 0.8;
maxSimilarityValue = 1;
xmin = 0.01;%0.01;
xmax = 0.1;%0.1; % 0.5;
filename = 'test_2_adaptive';
noise_min = 0; 
noise_max = 0.05;
%sparseness = 1-(log(N)/log(1.06))/N;
maxiter = 50;
useWorkers = true;
Data = struct;

xunstr=[1:8]; % koji atributi ulaze u kalkulacije za unstructurede
xsim= [9:10]; % koji atributi ulaze u kalkulacije za similarities


[y, similarity, R, x, alpha, beta, theta] = synthesize_data(...
    numTimeSteps,N,alpha, beta, sparseness,no_features, ...
    no_hiddenNeurons,minSimilarityValue,maxSimilarityValue, ...
    xmin, xmax,noise_min, noise_max,xsim);

save(filename)

%% train unstructured predictors
x_trening=x(:,xunstr,trainingTS);
y_trening=y(:,trainingTS);
x_tr = reshape(ipermute(x_trening,[1 3 2]),[],length(xunstr));
x_tr=[ones(size(x_tr,1),1) x_tr];

y_tr=y_trening(:);

Data.ThetaLR=  [inv(x_tr'*x_tr)*(x_tr'*y_tr)];%[1 ;

clearvars x_trening x_tr y_trening y_tr


%% calculate similarities

%% GCRF part


predictors{1}=R;

similaritiess = {};
for i = 1:size(R, 2)
   similaritiess{i} = similarity; 
end
similarities{1}=similaritiess;

                                      
ytr = y(:,trainingTS);
ytest = y(:,testcol);

lambdaAlpha = 0;
lambdaBeta = 0;
lambdaR=0;
lambdaS=0;
nalpha =length(predictors);
nbeta =length(similarities);
ntheta_r =length(xunstr);
ntheta_s = length(xsim);%  1;

tic
Data = createXCRFstruct( x, ytr, ytest, predictors, similarities, Data, xunstr, xsim,nalpha ,nbeta , ntheta_r,ntheta_s, lambdaAlpha, lambdaBeta,lambdaR, lambdaS, maxiter, useWorkers);

Data.u = trainCRFX(Data);
elapsedTime = toc

res=testCRFX(Data);

plotYgrid
mse_xcrf=mse(res,ytest)
mse_nn=mse(ytest,predictionNN)
r2=1- mse(ytest, res)/sum((ytest - mean(ytest)).^2)
r2_unstr=1- mse(ytest, predictionNN)/sum((ytest - mean(ytest)).^2)