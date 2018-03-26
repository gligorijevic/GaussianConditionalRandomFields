clc
clear;
close all

addpath('Plotting');
addpath('Predictors');
addpath('Similarity metrics');
addpath('Structure');
addpath('Synthetic data');
addpath('uncertaintyGCRF(uGCRF)');
addpath(genpath('GPdyn-new'));
addpath('util');
%% Setting parameters

use_all_data = false;
lag = 0;
trainTs = 26;
predictTs = 96;
select_features = [1,2,3,4,5,6]; %[1,2,3,4,5,6];
xunstr = 1:lag+numel(select_features); %12 / 18
xsim = 1:lag+numel(select_features); %[7,8];

monthsTr = lag+1:lag+trainTs;
monthsTest = lag+trainTs+1:lag+trainTs+predictTs;
T = [lag+1 lag+trainTs]; % parameter for original GCRF training
steps_to_predict = predictTs;

training_window = trainTs;
no_of_unstructured_predictors = 1;

useWorkers = true;
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
    benchmarkUncertaintyPropagationDirect(...
    xtrain, lagtrain, utrain, ytrain, xvalid, lagvalid, uvalid, yvalid, N,...
    lag, trainTs, predictTs, select_features, training_window);

%% Learn uGCRF
selectedConfidenceGuesses{1} = 0.95;
similarities_2{1} = {};
for nts = 1:trainTs+predictTs
   similarities_2{1}{nts} = similarities{1}; 
end
CRFData = createCRFstruct_uGCRF(...
    N, T, steps_to_predict, maxiter, y, X,...
    similarities_2, benchmark_predictors, benchmark_variances, ...
    selectedConfidenceGuesses);
CRFData.lambdaAlpha = 0*pi;
CRFData.lambdaBeta = 0*pi;

tic
[ualpha_new, ubeta_new]= trainCRF_new(CRFData, useWorkers);
toc

% Store optimized parameters in the structure

CRFData.ualpha_new = ualpha_new;
CRFData.ubeta_new = ubeta_new;

[predictionCRF_new, Sigma_new, Variance_new, predictionCRF_new_all, ...
    Sigma_new_all, Variance_new_all] = testCRF_new(CRFData);

for t = 1:trainTs+predictTs
    select_gcrf = N*(t-1)+1: t*N;
    benchmark_predictors{2}(:,t) = predictionCRF_new_all(select_gcrf);   
    benchmark_variances{2}(:,t) = Variance_new_all(select_gcrf);
end

%% Plot for the paper
addpath('paper plots');
node_idx = 6;

start_ts = 1;
end_ts = trainTs+predictTs;

start_ts_disp = 1;
end_ts_disp = trainTs+predictTs;
border_down = -0.5;
border_up = 3;

x_star = linspace(start_ts,end_ts,end_ts);
x_star = x_star(:);

% Defaults for this blog post
width = 5;     % Width in inches
height = 5;    % Height in inches
alw = 0.75;    % AxesLineWidth
fsz = 18;      % Fontsize
lw = 3;      % LineWidth
msz = 3;       % MarkerSize

gcrf_figure = figure();
% set(gcrf_figure, 'Visible', 'off');

pos = get(gcf, 'Position');
set(gcf, 'Position', [pos(1)+20 pos(2)+20 width*230, height*85]); %<- Set size
set(gca, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties

% subplot(2,1,1);

fill([x_star; flip(x_star,1)], [benchmark_predictors{2}(node_idx,start_ts_disp:end_ts_disp)' + ...
    1.96 * sqrt(benchmark_variances{2}(node_idx,start_ts_disp:end_ts_disp))'; ...
    flip(benchmark_predictors{2}(node_idx,start_ts_disp:end_ts_disp)' - ...
    1.96 * sqrt(benchmark_variances{2}(node_idx,start_ts_disp:end_ts_disp))', 1)], ...
    [7 7 7]/8 );
axis([-Inf inf border_down border_up])

% setting labels and font sizes
xlabel('Timesteps', 'FontSize', fsz,'FontWeight','bold');
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',18)
b = get(gca,'YTickLabel');
set(gca,'YTickLabel',b,'FontName','Times','fontsize',18)
% ylabel('Prediction with confidence interval');

hold on;
plot(start_ts:end_ts, benchmark_predictors{2}(node_idx,start_ts_disp:end_ts_disp), '-', 'MarkerFaceColor', 'r', 'LineWidth', lw);
plot(start_ts:end_ts, y(node_idx,lag+trainTs+start_ts:lag+trainTs+end_ts), '-', 'MarkerFaceColor', 'c', 'LineWidth', lw, 'MarkerSize', 1.5);
hold off;

% Save the file as PNG
set(gcf,'PaperPositionMode','auto')
print(sprintf('GCRF_precipitation_%i_direct_with_inputs.png', node_idx),'-dpng','-r0');

% subplot(2,1,2)
up_figure = figure();
% set(up_figure, 'Visible', 'off');
pos = get(gcf, 'Position');
set(gcf, 'Position', [pos(1)+20 pos(2)+20 width*230, height*85]); %<- Set size
set(gca, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties

fill([x_star; flip(x_star,1)], [benchmark_predictors{1}(node_idx,start_ts_disp:end_ts_disp)' + ...
    1.96 * sqrt(benchmark_variances{1}(node_idx,start_ts_disp:end_ts_disp))'; ...
    flip(benchmark_predictors{1}(node_idx,start_ts_disp:end_ts_disp)' - ...
    1.96 * sqrt(benchmark_variances{1}(node_idx,start_ts_disp:end_ts_disp))', 1)], ...
    [7 7 7]/8 );
axis([-Inf inf border_down border_up])

% setting labels and font sizes
xlabel('Timesteps', 'FontSize', fsz,'FontWeight','bold');
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',18)
b = get(gca,'YTickLabel');
set(gca,'YTickLabel',b,'FontName','Times','fontsize',18)
% ylabel('Prediction with confidence interval');

hold on;
plot(start_ts:end_ts, benchmark_predictors{1}(node_idx,start_ts_disp:end_ts_disp), '-', 'MarkerFaceColor', 'r', 'LineWidth', lw);
plot(start_ts:end_ts, y(node_idx,lag+trainTs+start_ts:lag+trainTs+end_ts), '-', 'MarkerFaceColor', 'c', 'LineWidth', lw, 'MarkerSize', 1.5);
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
print(sprintf('LR_precipitation_%i_direct_with_inputs.png', node_idx),'-dpng','-r0');

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

fprintf('========== Prediction performance ==========\n')
for pred = 1:numel(eval_prediction)
    results_mse(pred) = mse(yvalid, eval_prediction{pred});
end
fprintf('MSE results for multiple-steps-ahead are: [%f ; %f]\n',...
    results_mse);

for pred = 1:numel(eval_prediction)
    results_r2(pred) = 1 - ...
        (sum(sum((yvalid - eval_prediction{pred}).^2)))/...
        sum(sum((yvalid - mean(mean(yvalid))).^2));
end
fprintf('R2 results for multiple-steps-ahead are: [%f ; %f]\n',...
    results_r2);

for pred = 1:numel(eval_prediction)
    results_mse_osa(pred) = mse(yvalid(:,1), eval_prediction{pred}(:,1));
end
fprintf('MSE results for one-step-ahead are: [%f ; %f]\n',...
    results_mse_osa);


for pred = 1:numel(eval_prediction)
    results_r2_osa(pred) = 1 - ...
        (sum(sum((yvalid(:,1) - eval_prediction{pred}(:,1)).^2)))/...
        sum(sum((yvalid(:,1) - mean(mean(yvalid(:,1)))).^2));
end
fprintf('R2 results for one-step-ahead are: [%f ; %f]\n',...
    results_r2_osa);
fprintf('============================================\n')

%% Test uncertainty estimation performance of the models
fprintf('========== Uncertainty estimation performance ==========\n')
for pred = 1:numel(eval_prediction)
    results_nlpd(pred) = ...
        mean(mean(1/size(y,1) .* ((yvalid - eval_prediction{pred}).^2./...
        (2 .* eval_variance{pred}) + log((eval_variance{pred})))));
    results_nlpd_optim(pred) = ...
        mean(mean(1/size(y,1) .* ((yvalid - eval_prediction{pred}).^2./...
        (2 .* eval_opt_variance{pred}) + log((eval_opt_variance{pred})))));
end
fprintf('NLPD results for multiple-steps-ahead are: [%f ; %f]\n',...
    results_nlpd);
fprintf('Optimal NLPD results for multiple-steps-ahead are: [%f ; %f]\n',...
    results_nlpd_optim);

for pred = 1:numel(eval_prediction)
    results_nlpd_osa(pred) = ...
        mean(1/size(y,1) .* ((yvalid(:,1) - eval_prediction{pred}(:,1)).^2./...
        (2 .* eval_variance{pred}(:,1)) + log((eval_variance{pred}(:,1)))));
    results_nlpd_optim_osa(pred) = ...
        mean(1/size(y,1) .* ((yvalid(:,1) - eval_prediction{pred}(:,1)).^2./...
        (2 .* eval_opt_variance{pred}(:,1))...
        + log((eval_opt_variance{pred}(:,1)))));
end
fprintf('NLPD results for one-step-ahead are: [%f ; %f]\n',...
    results_nlpd_osa);
fprintf('Optimal NLPD results for one-step-ahead are: [%f ; %f]\n',...
    results_nlpd_optim_osa);
fprintf('========================================================\n')

