function [ predictors, variances] = ...
    benchmarkUncertaintyPropagationIterative( ...
    xtrain, lagtrain, utrain, ytrain, xvalid, lagvalid, uvalid, yvalid,...
    N, lag, trainTs, predictTs, select_features, training_window)

R_linreg = NaN(N, trainTs+predictTs);
Var_linreg = NaN(N, trainTs+predictTs);
R_gp = NaN(N, trainTs+predictTs);
Var_gp = NaN(N, trainTs+predictTs);

%% Define covariance function: Squared exponential covariance function with
% ARD (For Gaussian Processes)
covfunc = @covSEard;

% Define covariance function: Gaussinan likelihood function.
likfunc = @likGauss;

% Define mean function: Zero mean function.
meanfunc = @meanZero;

% Define inference method: Exact Inference
inffunc = @infExact;

%% Setting initial hyperparameters (For Gaussian Processes)

D = size(xtrain,3); % Input space dimension
hyp.cov  = ones(D+1,1); %-ones(D+1,1);

% Define likelihood hyperparameter. In our case this parameter is noise
% parameter.
hyp.lik = log(0.1);

hyp.mean = [];
% hyp.mean = hypsu;

%% Start for loop for each node in the graph
for i = 1:N
    %% Prepare training and testing inputs
    xx_train = squeeze(xtrain(i,1:trainTs,:));
    yy_train = ytrain(i,1:trainTs);
    
    %%  Train Linear ARX
    linear_mdl = LinearModel.fit(xx_train, yy_train);
    % Predict using Linear ARX
    R_linreg(i,1:trainTs) = predict(linear_mdl, xx_train);
    
    for lrTs = 1:trainTs
        Var_linreg(i,lrTs) = ...
            sum((yy_train(lrTs) - predict(linear_mdl, xx_train(lrTs,:))).^2)/...
            (size(xx_train, 1) - size(xx_train, 2));
    end
    %% Traing Gaussian Processes, Predict and calculate variance
    hyp = ...
        trainGParx(hyp, inffunc, meanfunc, covfunc, likfunc,...
        xx_train, yy_train');

%     hyp = ...
%         minimize(hyp, @gp, -100, inffunc, meanfunc, covfunc, likfunc, ...
%         xx_train, yy_train');
    
    % Predict on training data to obtain residuals
    [R_gp(i,1:trainTs), Var_gp(i,1:trainTs)] = ...
        gp(hyp, inffunc, meanfunc, covfunc, likfunc,...
        xx_train, yy_train', xx_train);
    
    %% Iterative-one-step-ahead prediction for Linear Regression
    
    % First predict one-step-ahead to obtain first prediction in the future
    xx_test = squeeze(xvalid(i,1,:))';
    
    R_linreg(i, trainTs+1) = predict(linear_mdl, xx_test);
    Var_linreg(i, trainTs+1) = ...
        Var_linreg(i, trainTs) .* ...
        diag(1 + xx_test/(xx_train' * xx_train) * xx_test');
    
    for ntsPredict = 2:predictTs

        xx_test = squeeze(xvalid(i,ntsPredict,:))';
        xx_test = ...
            [xx_test(2:lag) R_linreg(i, trainTs+ntsPredict-1)...
            xx_test(lag+1:end)];

        R_linreg(i, trainTs+ntsPredict) = predict(linear_mdl, xx_test);
        Var_linreg(i, trainTs+ntsPredict) = ...
            Var_linreg(i, trainTs+ntsPredict-1) .*...
            diag(1 + xx_test/(xx_train' * xx_train) * xx_test');

    end
    
    %% Iterative-one-step-ahead prediction with Taylor approximation GP
    xx_test = squeeze(xvalid(i,1:predictTs,:));
    
    [R_gp(i,trainTs+1:end), Var_gp(i,trainTs+1:end)] = ...
        simulGPtaylorSE(hyp, inffunc, meanfunc, covfunc, likfunc,...
        xx_train, yy_train', xx_test, lag);
end

predictors{1} = R_linreg;
predictors{2} = R_gp;
variances{1} = Var_linreg;
variances{2} = Var_gp;

end

