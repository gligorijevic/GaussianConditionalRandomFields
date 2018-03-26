clc
clear;
close all

addpath('Plotting');
addpath('Predictors');
addpath('Similarity metrics');
addpath('Structure');
addpath('Synthetic data');
addpath('adaptiveGCRF');
addpath('GCRF');
addpath('fastExactGCRF');
addpath(genpath('GPdyn-new'));
addpath('Util');
%% Setting parameters

use_all_data = false;
% lag = how monay previous time step values are used as inputs 
lag = 12;
% trainTs = number of time steps used for training the model
trainTs = 50;
% predictTs = number of time steps used for predicting ahead
predictTs = 96;
% [] means we are not using any of the input variables available
select_features = []; %[1,2,3,4,5,6];
xunstr = 1:lag+numel(select_features); %12 / 18
xsim = 1:lag+numel(select_features); %[7,8];

% Previous parameters stored for conveniece purposes
monthsTr = lag+1:lag+trainTs;
monthsTest = lag+trainTs+1:lag+trainTs+predictTs;
T = [lag+1 lag+trainTs];
steps_to_predict = predictTs;

training_window = trainTs;
no_of_unstructured_predictors = 1;

useWorkers = true; % use MATLAB paralelization toolbox
maxiter = 50; % maximum number of iterations for optimization function
%% Get rain data
load data/rain_data_northwest.mat

clearvars -except use_all_data lag trainTs predictTs monthsTr ...
    monthsTest T steps_to_predict numTimeSteps training_window ...
    select_features no_of_unstructured_predictors xunstr xsim ...
    useWorkers maxiter N X y similarities

%% Prepare the data
[xtrain, lagtrain, utrain, ytrain, xvalid, lagvalid, uvalid, yvalid] = ...
    prepareDataNoisyInputs(...
    X, y, N, select_features, lag, trainTs, predictTs);

%% Benchmarks
N = size(ytrain, 1);
[benchmark_predictors, benchmark_variances] = ...
    benchmarkUncertaintyPropagationIterative(...
    xtrain, lagtrain, utrain, ytrain, xvalid, lagvalid, uvalid, yvalid, N,...
    lag, trainTs, predictTs, select_features, training_window);

%% Prepare unstructured predictor (Linear model)
xx = []; yy = [];
for nts = 1:trainTs
    xx = [xx; squeeze(xtrain(:,nts,:))];
    yy = [yy; ytrain(:,nts)];
end

mdl = LinearModel.fit(xx, yy);
theta_init = mdl.Coefficients.Estimate';

predictors{1} = NaN(size(ytrain,1),size(ytrain,2)+1);
for nts = 1:trainTs
    predictors{1}(:,nts) = predict(mdl, squeeze(xtrain(:,nts,:)));
end

% Predict one step ahead
predictors{1}(:,trainTs+1) = predict(mdl, squeeze(xvalid(:,1,:)));

for nts = 1:predictTs
    predictors{1}(:,trainTs+nts) = predict(mdl, squeeze(xvalid(:,nts,:)));
end

%% Prepare similarity metric (Initialize Gaussian kernel parameters by 
% Gaussian Processes regression optimization)
%shrink the data to speed-up training of gaussian processes
input = xx(1:500,:);
target = yy(1:500);

cov = @covSEard;
% Define covariance function: Gaussinan likelihood function.
lik = @likGauss;
% Define mean function: Zero mean function.
meanFunc = @meanZero;
% meanFunc = {'meanSum',{@meanZero,@meanConst}};  hypsu = [round(mean(ytrain))]; % sum
% Define inference method: Exact Inference
inf= @infExact;
% Setting initial hyperparameters
D = size(input,2); % Input space dimension
hyp.cov  = -ones(D+1,1);
% Define likelihood hyperparameter. In our case this parameter is noise
% parameter.
hyp.lik = log(0.1);
hyp.mean = [];

% Training
% Identification of the GP model
[hyp, flogtheta, i] =...
    trainGParx(hyp, inf, meanFunc, cov, lik, input, target);

psi_init = hyp.cov';

% Calculate similarity matrix for first timestep based on Gaussian Kernel
if exist('similarties','var')
    [xx, yy] = meshgrid(1:N,1:N);
    X_sim = squeeze(xtrain(:,1,:));
    X_sim_dist_sq = (X_sim(xx,:)-X_sim(yy,:)).^2;
    temp_sum = bsxfun(@rdivide, X_sim_dist_sq, 2*(psi_init(2:end).^2));
    similarities{1} = psi_init(1)*reshape(exp(-sum(temp_sum, 2)),N,N);
end

clearvars xx yy input target mdl GPData cov lik hyp meanFunc inf D ...
    X_sim X_sim_dist_sq temp_sum flogtheta i nts

%% Learn original GCRF (Radosavljevic et al. 2010)
CRFData = createCRFstruct(...
    N, T, steps_to_predict, maxiter, y, X, similarities, predictors);
CRFData.lambdaAlpha = 0*pi;
CRFData.lambdaBeta = 0*pi;

tic
[ualpha, ubeta]= trainCRFFast(CRFData, useWorkers);
elapsedTimeLinGCRF = toc
alpha_init = ualpha;
beta_init = ubeta;
CRFData.ualpha = ualpha;
CRFData.ubeta = ubeta;

% Testing GCRF models
[predictionCRF, Sigma, Variance,...
    predictionCRF_all, Sigma_all, Variance_all] = testCRFFast(CRFData);

%% Learn new adaptive GCRF

% Regularization parameters
lambdaAlpha = 0;
lambdaBeta = 0;
lambdaR = 0;
lambdaS = 0;

% Initialization of parameters
thetaAlpha = alpha_init;
thetaBeta = beta_init;
thetaR = theta_init;
thetaS = psi_init;


nalpha = length(predictors);
nbeta = length(similarities);
ntheta_r = length(xunstr) + 1; % add one for the bias term
ntheta_s = length(xsim) + 1; % add one for the outer parameter

Data = struct;
xinput = cat(3, ipermute(xtrain,[1,3,2]), ipermute(xvalid, [1,3,2]));

tic
% Create structure to carry information during training process
Data = createXCRFstruct( xinput, ytrain, yvalid, predictors, similarities, Data, ...
    xunstr, xsim, nalpha, nbeta, ntheta_r, ntheta_s, lambdaAlpha, lambdaBeta,...
    lambdaR, lambdaS, thetaAlpha, thetaBeta, thetaR, thetaS, lag, maxiter, useWorkers);
% Train model
Data.u = trainCRFX(Data);
elapsedTimeAdaptGCRF = toc

clearvars lambdaAlpha lambdaBeta lambdaR lambdaS thetaAlpha  ...
    thetaBeta  thetaR  thetaS nalpha nbeta ...
    ntheta_r ntheta_s

%% Taylor simulation of the GCRF prediction
tic
% Apply multiple-steps-ahead prediction with Taylor simulation of first
% two moments of the distribution
[muNoisyGCRF, sigmaNoisyGCRF, SigmaX] = simulGCRFTaylor(Data);
elapsedTimePredictionPropagation = toc

% save workspace
% save('precipitation_propagation_iterative_lag_12_results.mat','-v7.3');

%% Plot uncertainty
node_idx = 1;

start_ts = 1;
end_ts = Data.Ttr+Data.Ttest;

x_star = linspace(start_ts,end_ts,end_ts);
x_star = x_star(:);

% Defaults for this blog post
width = 5;     % Width in inches
height = 5;    % Height in inches
alw = 0.75;    % AxesLineWidth
fsz = 9;      % Fontsize
lw = 1.5;      % LineWidth
msz = 3;       % MarkerSize

figure

pos = get(gcf, 'Position');
% set(gcf, 'Position', [pos(1)+40 pos(2)+40 width*100, height*100]); %<- Set size
set(gca, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties


subplot(3,1,1);

fill([x_star; flip(x_star,1)], [muNoisyGCRF(node_idx,:)' + ...
    1.96 * sqrt(squeeze(sigmaNoisyGCRF(node_idx,node_idx,:))); ...
    flip(muNoisyGCRF(node_idx,:)' - ...
    1.96 * sqrt(squeeze(sigmaNoisyGCRF(node_idx,node_idx,:))), 1)], ...
    [7 7 7]/8 );
xlabel('Timesteps');
ylabel('Prediction with confidence interval');
hold on;
plot(start_ts:end_ts, muNoisyGCRF(node_idx,:), 'ro--', 'LineWidth', 1.5);
plot(start_ts:end_ts, y(node_idx,Data.lag+start_ts:Data.lag+end_ts), '.-', 'MarkerFaceColor', 'c', 'LineWidth', 1.5, 'MarkerSize', 10);
hold off;
hleg1 = legend('uncertainty region', 'predicted values', 'true values');
set(hleg1,'Location','NorthWest');
set(hleg1,'Interpreter','none');
title('GCRF with noisy inputs');

% subplot(3,1,2)
% fill([x_star; flip(x_star,1)], [benchmark_predictors{1}(node_idx,:) + ...
%     1.96 * sqrt(benchmark_variances{1}(node_idx,:)), ...
%     flip(benchmark_predictors{1}(node_idx,:) - ...
%     1.96 * sqrt(benchmark_variances{1}(node_idx,:)), 1)], ...
%     [7 7 7]/8 );
% xlabel('Timesteps');
% ylabel('Prediction with confidence interval');
% hold on;
% plot(start_ts:end_ts, benchmark_predictors{1}(node_idx,:), 'ro--', 'LineWidth', 1.5);
% plot(start_ts:end_ts, y(node_idx,Data.lag+start_ts:Data.lag+end_ts), '.-', 'MarkerFaceColor', 'c', 'LineWidth', 1.5, 'MarkerSize', 10);
% hold off;
% hleg1 = legend('uncertainty region', 'predicted values', 'true values');
% set(hleg1,'Location','NorthWest');
% set(hleg1,'Interpreter','none');
% title('Linear Regression with noisy inputs');

subplot(2,1,2)
% fill([x_star; flip(x_star,1)], [muNoisyGCRF(node_idx,:)' + ...
%     1.96 * sqrt(squeeze(sigmaNoisyGCRF(node_idx,node_idx,:))); ...
%     flip(muNoisyGCRF(node_idx,:)' - ...
%     1.96 * sqrt(squeeze(sigmaNoisyGCRF(node_idx,node_idx,:))), 1)], ...
%     [7 7 7]/8 );
fill([x_star; flip(x_star,1)], [benchmark_predictors{2}(node_idx,:)' + ...
    1.96 * sqrt(benchmark_variances{2}(node_idx,:))'; ...
    flip(benchmark_predictors{2}(node_idx,:)' - ...
    1.96 * sqrt(benchmark_variances{2}(node_idx,:))', 1)], ...
    [7 7 7]/8 );
xlabel('Timesteps');
ylabel('Prediction with confidence interval');
hold on;
plot(start_ts:end_ts, benchmark_predictors{2}(node_idx,:), 'ro--', 'LineWidth', 1.5);
plot(start_ts:end_ts, y(node_idx,Data.lag+start_ts:Data.lag+end_ts), '.-', 'MarkerFaceColor', 'c', 'LineWidth', 1.5, 'MarkerSize', 10);
hold off;
hleg1 = legend('uncertainty region', 'predicted values', 'true values');
set(hleg1,'Location','NorthWest');
set(hleg1,'Interpreter','none');
title('Gaussian Processes with noisy inputs');

clearvars node_idx start_ts end_ts x_star width height alw fsz lw msz hleg1 pos

%% Plot for the paper
addpath('paper plots');
node_idx = 1;

start_ts = 1;
end_ts = Data.Ttest-2;

start_ts_disp = Data.Ttr+3;
end_ts_disp = Data.Ttr+Data.Ttest;
border = 5;

% a=sum((muNoisyGCRF(:,start_ts:end_ts) - y(:,Data.lag+start_ts:Data.lag+end_ts)).^2,2);
% 
% [B, I]=sort(a);
% I(4)

x_star = linspace(start_ts,end_ts,end_ts);
x_star = x_star(:);

% Defaults for this blog post
width = 5;     % Width in inches
height = 5;    % Height in inches
alw = 0.75;    % AxesLineWidth
fsz = 24;      % Fontsize
lw = 2;      % LineWidth
msz = 3;       % MarkerSize

gcrf_figure = figure();
% set(gcrf_figure, 'Visible', 'off');

pos = get(gcf, 'Position');
set(gcf, 'Position', [pos(1)+20 pos(2)+20 width*230, height*85]); %<- Set size
set(gca, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties

% subplot(2,1,1);

fill([x_star; flip(x_star,1)], [muNoisyGCRF(node_idx,start_ts_disp:end_ts_disp)' + ...
    1.96 * sqrt(squeeze(sigmaNoisyGCRF(node_idx,node_idx,start_ts_disp:end_ts_disp))); ...
    flip(muNoisyGCRF(node_idx,start_ts_disp:end_ts_disp)' - ...
    1.96 * sqrt(squeeze(sigmaNoisyGCRF(node_idx,node_idx,start_ts_disp:end_ts_disp))), 1)], ...
    [7 7 7]/8 );
axis([-Inf inf -border border])

% setting labels and font sizes
xlabel('Timesteps', 'FontSize', fsz,'FontWeight','bold');
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',20)
b = get(gca,'YTickLabel');
set(gca,'YTickLabel',b,'FontName','Times','fontsize',20)
% ylabel('Prediction with confidence interval');

hold on;
plot(start_ts:end_ts, muNoisyGCRF(node_idx,start_ts_disp:end_ts_disp), '-', 'MarkerFaceColor', 'r', 'LineWidth', lw);
plot(start_ts:end_ts, y(node_idx,Data.lag+Data.Ttr+start_ts:Data.lag+Data.Ttr+end_ts), '-', 'MarkerFaceColor', 'c', 'LineWidth', lw, 'MarkerSize', 1.5);
hold off;
%hleg1 = legend('uncertainty region', 'predicted values', 'true values');
%set(hleg1,'Location','NorthWest');
%set(hleg1,'Interpreter','none');
% title('Gaussian Conditional Random Fields');

% Here we preserve the size of the image when we save it.
% set(gcf,'InvertHardcopy','on');
% set(gcf,'PaperUnits', 'inches');
% papersize = get(gcf, 'PaperSize');
% left = (papersize(1)- width)/2;
% bottom = (papersize(2)- height)/2;
% myfiguresize = [left, bottom, width, height];
% set(gcf,'PaperPosition', myfiguresize);

% Save the file as PNG
set(gcf,'PaperPositionMode','auto')
print(sprintf('GCRF_precipitation_%i_iterative.png', node_idx),'-dpng','-r0');

% subplot(2,1,2)
up_figure = figure();
% set(up_figure, 'Visible', 'off');
pos = get(gcf, 'Position');
set(gcf, 'Position', [pos(1)+20 pos(2)+20 width*230, height*85]); %<- Set size
set(gca, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties

fill([x_star; flip(x_star,1)], [benchmark_predictors{2}(node_idx,start_ts_disp:end_ts_disp)' + ...
    1.96 * sqrt(benchmark_variances{2}(node_idx,start_ts_disp:end_ts_disp))'; ...
    flip(benchmark_predictors{2}(node_idx,start_ts_disp:end_ts_disp)' - ...
    1.96 * sqrt(benchmark_variances{2}(node_idx,start_ts_disp:end_ts_disp))', 1)], ...
    [7 7 7]/8 );
axis([-Inf inf -border border])

% setting labels and font sizes
xlabel('Timesteps', 'FontSize', fsz,'FontWeight','bold');
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',20)
b = get(gca,'YTickLabel');
set(gca,'YTickLabel',b,'FontName','Times','fontsize',20)

% ylabel('Prediction with confidence interval');
hold on;
plot(start_ts:end_ts, benchmark_predictors{2}(node_idx,start_ts_disp:end_ts_disp), '-', 'MarkerFaceColor', 'r', 'LineWidth', lw);
plot(start_ts:end_ts, y(node_idx,Data.lag+Data.Ttr+start_ts:Data.lag+Data.Ttr+end_ts), '-', 'MarkerFaceColor', 'c', 'LineWidth', lw, 'MarkerSize', 1.5);
hold off;
% title('Gaussian Processes Regression');

% Here we preserve the size of the image when we save it.
set(gcf,'InvertHardcopy','on');
set(gcf,'PaperUnits', 'inches');
papersize = get(gcf, 'PaperSize');
left = (papersize(1)- width)/2;
bottom = (papersize(2)- height)/2;
myfiguresize = [left, bottom, width, height];
set(gcf,'PaperPosition', myfiguresize);

% Save the file as PNG
set(gcf,'PaperPositionMode','auto')
print(sprintf('GP_precipitation_%i_iterative.png', node_idx),'-dpng','-r0');

clearvars node_idx start_ts end_ts x_star width height alw fsz lw msz hleg1 pos


%% Test performance of the models
eval_prediction = {};
eval_variance = {};
eval_opt_variance = {};
for pred = 1:numel(benchmark_predictors)
    eval_prediction{pred} = benchmark_predictors{pred}(:,trainTs+1:end);
    eval_variance{pred} = benchmark_variances{pred}(:,trainTs+1:end);
    eval_opt_variance{pred} = ...
        (yvalid - benchmark_predictors{pred}(:,trainTs+1:end)).^2;
end
eval_prediction{3} = muNoisyGCRF(:,trainTs+1:end);
for i = 1: N
    eval_variance{3}(i,:) = sigmaNoisyGCRF(i,i,trainTs+1:end);
end
eval_opt_variance{3} = ...
    (yvalid - muNoisyGCRF(:,trainTs+1:end)).^2;



fprintf('========== Prediction performance ==========\n')
for pred = 1:numel(eval_prediction)
    results_mse(pred) = mse(yvalid, eval_prediction{pred});
end
fprintf('MSE results for multiple-steps-ahead are: [%f ; %f ; %f]\n',...
    results_mse);

for pred = 1:numel(eval_prediction)
    results_mse_osa(pred) = mse(yvalid(:,1), eval_prediction{pred}(:,1));
end
fprintf('MSE results for one-step-ahead are: [%f ; %f ; %f]\n',...
    results_mse_osa);

fprintf('============================================\n')
bar([1,2,3],results_mse, results_mse_osa,'FaceColor',[0 .5 .5],'EdgeColor',[0 .9 .9],'LineWidth',1.5);

