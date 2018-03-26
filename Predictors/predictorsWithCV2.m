function [ predictors, variances, Weights ] = predictorsWithCV2( x, xGhost, y, yGhost, numTimeSteps, N, T, training_window )
% Generates predictions using several predictors
% Linear Autoregressive model
% Gaussian Processes Regression
% Lasso regression
% Elastic Net regression
% Neural Networks

% warning off;

addpath(genpath('..\gpml-matlab-v3.4-2013-11-11\'));

meanfunc = {@meanZero};
M = 8;
covfunc = {@covSEard};
L = ones(M, 1); sf = 1; hyp0.cov = log([L; sf]);
likfunc = @likGauss; sn = 5; hyp0.lik = log(sn);
inffunc = @infExact;

%Neural network
net = fitnet(6);

R_linreg_lag0 = zeros(N, numTimeSteps);
Var_linreg_lag0 = zeros(N, numTimeSteps);

R_lasso_lag0 = zeros(N, numTimeSteps);
Var_lasso_lag0 = zeros(N, numTimeSteps);

R_elasticnet_lag0 = zeros(N, numTimeSteps);
Var_elasticnet_lag0 = zeros(N, numTimeSteps);

R_gp_lag0 = zeros(N, numTimeSteps);
Var_gp_lag0 = zeros(N, numTimeSteps);

R_NN_lag0 = zeros(N, numTimeSteps);

weights_linreg_lag0 = zeros(N, M+1);
weights_lasso_lag0 = zeros(N, M+1);
weights_elasticnet_lag0 = zeros(N, M+1);

% Training models



%% K-fold Cross Validation

fold = 4;

xi = x(:,:,T(1): T(2));
xi_test=x(:,:, T(2)-1:end);

xxi = reshape(ipermute(xi,[1 3 2]),[],size(x,2));
xxi_test=reshape(ipermute(xi_test,[1 3 2]),[],size(x,2));

xxGhost = xGhost(:,:,:);
xxGhost = reshape(ipermute(xxGhost,[1 3 2]),[],size(xxGhost,2));

yi=y(:,T(1):T(2))';
yi = yi(:);
for iterval=1:fold
    
    iterval;
    tresh_down = round((iterval-1) * (1/fold) * T(2) * N + 1);
    tresh_up = round(iterval * (1/fold) * T(2) * N);
    
    %Training X and Y
    testx=xxi(tresh_down:tresh_up,:);
    yy_cv_test = yi(tresh_down:tresh_up);
    
    trening= [xxGhost; xxi(1:(tresh_down-1),:)];
    ytre=[yGhost(:,:)'; yi(1:(tresh_down-1))];
    
    trening = trening(end-training_window*N+1:end,:);
    ytre = ytre(end-training_window*N+1:end);
    
    duzina = size(trening, 1);
    xx_lag0 = trening;
    yy_lag0 = ytre;
    
    xx_lag0_cv_test = testx(1:end,:);
    
    % Train Linear ARX
    mdl_lag0 = LinearModel.fit(xx_lag0, yy_lag0);
    weights_lag0 = linearWeights(yy_lag0);
    
    % Train Lasso
    [B_lasso_lag0, FitInfo_lasso_lag0] = lasso(xx_lag0, yy_lag0, 'Alpha', 1, 'DFmax', 4, 'NumLambda', 100);
    
    % Train Elastic Net
    [B_elnet_lag0, FitInfo_elnet_lag0] = lasso(xx_lag0, yy_lag0,'Alpha',0.5, 'DFmax', 4,'NumLambda', 100);
    
    % Predict using Linear ARX
    R_linreg_lag0(:,tresh_down:tresh_up) = predict(mdl_lag0, xx_lag0_cv_test);
    
    % Predict using Lasso
    B_lasso_lag0 = selectLambdaLasso(B_lasso_lag0, FitInfo_lasso_lag0);
    R_lasso_lag0(:,tresh_down:tresh_up) = glmval(B_lasso_lag0, xx_lag0_cv_test, 'identity');
    
    % Predict using Elastic Net
    B_elnet_lag0 = selectLambdaLasso(B_elnet_lag0, FitInfo_elnet_lag0);
    R_elasticnet_lag0(:,tresh_down:tresh_up) = glmval(B_elnet_lag0, xx_lag0_cv_test, 'identity');
    
    % Traing Gaussian Processes, Predict and calculate variance
    hyp0 = minimize(hyp0, @gp, -100, inffunc, meanfunc, covfunc, likfunc, xx_lag0, yy_lag0);
    
    R_gp_lag0(:,tresh_down:tresh_up) = ...
        gp(hyp0, inffunc, meanfunc, covfunc, likfunc, xx_lag0, yy_lag0, xx_lag0_cv_test);
    
    [net,tr] = train(net,xx_lag0',yy_lag0');
    nntraintool
    R_NN_lag0(:,tresh_down:tresh_up) = net(xx_lag0_cv_test');
    
end

trening= xxi(:,:);
ytre= yi(:);
ytre = ytre(end-training_window+1:end);
trening = trening(end-training_window+1:end,:);

ytest =  y(:,:)';
ytest = ytest(T(2)-1:end);

duzina = size(trening, 1);
xx_lag0 = trening;
yy_lag0 = ytre;

xx_lag0_test = xxi_test(3:end,:);

% Train Linear ARX
mdl_lag0 = LinearModel.fit(xx_lag0, yy_lag0);

%     weights_lag0 = linearWeights(yy_lag0);

% Train Linear ARX
[B_lasso_lag0, FitInfo_lasso_lag0] = lasso(xx_lag0, yy_lag0, 'Alpha', 1, 'DFmax', 4, 'NumLambda', 100);

% Train Linear ARX
[B_elnet_lag0, FitInfo_elnet_lag0] = lasso(xx_lag0, yy_lag0, 'Alpha', 0.5, 'DFmax', 4, 'NumLambda', 100);

% Predictive weights for Lasso
B_lasso_lag0 = selectLambdaLasso(B_lasso_lag0, FitInfo_lasso_lag0);

% Predictive weights for Elastic Net
B_elnet_lag0 = selectLambdaLasso(B_elnet_lag0, FitInfo_elnet_lag0);

weights_linreg_lag0(:, :) = mdl_lag0.Coefficients.Estimate';

weights_lasso_lag0(:, :) = B_lasso_lag0';

weights_elasticnet_lag0(:, :) = B_elnet_lag0';

%Fix uncertainty estimation now after CV is done on a node
for j = T(2)+1:numTimeSteps
    
    xxx_lag0 = xx_lag0_test(j-T(2),:);
    
    % Predict using Linear ARX
    R_linreg_lag0(:,j) = predict(mdl_lag0, xxx_lag0);
    
    % Predict using Lasso
    R_lasso_lag0(:,j) = glmval(B_lasso_lag0, xxx_lag0, 'identity');
    
    % Predict using Elastic Net
    R_elasticnet_lag0(:,j) = glmval(B_elnet_lag0, xxx_lag0, 'identity');
    
end

% Traing Gaussian Processes, Predict and calculate variance
hyp0 = minimize(hyp0, @gp, -100, inffunc, meanfunc, covfunc, likfunc, xx_lag0, yy_lag0);

R_gp_lag0(:,T(2)+1:end) = ...
    gp(hyp0, inffunc, meanfunc, covfunc, likfunc,...
    xx_lag0, yy_lag0, xx_lag0_test);

[net,tr] = train(net,xx_lag0',yy_lag0');
nntraintool
R_NN_lag0(:,T(2)+1:end) = net(xx_lag0_test');

predictors{1} = R_linreg_lag0;

predictors{2} = R_lasso_lag0;

predictors{3} = R_elasticnet_lag0;

predictors{4} = R_gp_lag0;

predictors{5} = R_NN_lag0;

Weights{1} = weights_linreg_lag0;
Weights{2} = weights_lasso_lag0;
Weights{3} = weights_elasticnet_lag0;

end