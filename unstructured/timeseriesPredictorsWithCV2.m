function [ predictors, variances, confidence_guesses, Weights ] = timeseriesPredictorsWithCV2( x, xGhost, y, yGhost, numTimeSteps, N, T, training_window )
% Generates predictions using several predictors
% Linear Autoregressive model with lag0,lag1,lag2
% Gaussian Processes Regression with lag0,lag1,lag2
% Lasso regression with lag0,lag1,lag2
% Elastic Net regression with lag0,lag1,lag2

warning off;

addpath(genpath('..\gpml-matlab-v3.4-2013-11-11\'));

% meanfunc = {@meanSum, {@meanLinear, @meanConst}}; 
% L = rand(size(x, 2), 1); hyp0.mean = [L; 1];
% L = rand(size(x, 2)*2+1, 1); hyp1.mean = [L; 1]; 
% L = rand(size(x, 2)*3+2, 1); hyp2.mean = [L; 1];
meanfunc = {@meanZero};
% hyp0.cov = log([ell; sf]); hyp1.cov = log([ell; sf]); hyp2.cov = log([ell; sf]);
% covfunc = {@covPoly,2}; c = 2; sf = 1; 
% hyp0.c% covfunc = {@covMaterniso, 3}; ell = 2000; sf = 2000; 
% hyp0.cov = log([c;sf]); hyp1.cov = log([c;sf]); hyp2.cov = log([c;sf]);

covfunc = {@covSEard};
L = ones(size(x, 2), 1); sf = 1; hyp0.cov = log([L; sf]);
L = ones(size(x, 2)*2+1, 1); sf = 1; hyp1.cov = log([L; sf]); %adding one for y from previous timestep
L = ones(size(x, 2)*3+2, 1); sf = 1; hyp2.cov = log([L; sf]); %adding two for y from previous timesteps
likfunc = @likGauss; sn = 5; hyp0.lik = log(sn); hyp1.lik = log(sn); hyp2.lik = log(sn);
% covfunc = @covSEiso; hyp0.cov = [0; 0];hyp1.cov = [0; 0];hyp2.cov = [0; 0];
% hyp.cov = [0; 0]; hyp.mean = [0; 0]; hyp.lik = log(0.1);
% hyp = minimize(hyp, @gp, -100, @infExact, meanfunc, covfunc, likfunc, x, y);
% [m s2] = gp(hyp, @infExact, meanfunc, covfunc, likfunc, x, y, z);






R_linreg_lag0 = zeros(N, numTimeSteps);
R_linreg_lag1 = zeros(N, numTimeSteps);
R_linreg_lag2 = zeros(N, numTimeSteps);
Var_linreg_lag0 = zeros(N, numTimeSteps);
Var_linreg_lag1 = zeros(N, numTimeSteps);
Var_linreg_lag2 = zeros(N, numTimeSteps);

R_lasso_lag0 = zeros(N, numTimeSteps);
R_lasso_lag1 = zeros(N, numTimeSteps);
R_lasso_lag2 = zeros(N, numTimeSteps);
Var_lasso_lag0 = zeros(N, numTimeSteps);
Var_lasso_lag1 = zeros(N, numTimeSteps);
Var_lasso_lag2 = zeros(N, numTimeSteps);

R_elasticnet_lag0 = zeros(N, numTimeSteps);
R_elasticnet_lag1 = zeros(N, numTimeSteps);
R_elasticnet_lag2 = zeros(N, numTimeSteps);
Var_elasticnet_lag0 = zeros(N, numTimeSteps);
Var_elasticnet_lag1 = zeros(N, numTimeSteps);
Var_elasticnet_lag2 = zeros(N, numTimeSteps);

R_gp_lag0 = zeros(N, numTimeSteps);
R_gp_lag1 = zeros(N, numTimeSteps);
R_gp_lag2 = zeros(N, numTimeSteps);
Var_gp_lag0 = zeros(N, numTimeSteps);
Var_gp_lag1 = zeros(N, numTimeSteps);
Var_gp_lag2 = zeros(N, numTimeSteps);

weights_linreg_lag0 = zeros(N, 9);
weights_linreg_lag1 = zeros(N, 18);
weights_linreg_lag2 = zeros(N, 27);
weights_lasso_lag0 = zeros(N, 9);
weights_lasso_lag1 = zeros(N, 18);
weights_lasso_lag2 = zeros(N, 27);
weights_elasticnet_lag0 = zeros(N, 9);
weights_elasticnet_lag1 = zeros(N, 18);
weights_elasticnet_lag2 = zeros(N, 27);


%% K-fold Cross Validation

for i = 1:N
    fold = 10;
    
    xi = x(i,:,T(1): T(2));
    xi_test=x(i,:, T(2)-1:end);
    xxi = reshape(ipermute(xi,[1 3 2]),[],size(x,2));
    xxi_test=reshape(ipermute(xi_test,[1 3 2]),[],size(x,2));
    yi=y(i,T(1):T(2))';
    for iterval=1:fold
        
        iterval;
        tresh_down = round((iterval-1) * (1/fold) * T(2) + 1);
        tresh_up = round(iterval * (1/fold) * T(2));
        
        disp(['Node: ', num2str(i),' is on ',num2str(iterval),' step of crossvalidation, data is taking steps from: ',num2str(tresh_down),' to: ',num2str(tresh_up)]);
        
        %Training X and Y
        testx=xxi(tresh_down:tresh_up,:);
        yy_cv_test = yi(tresh_down:tresh_up);
        
        if tresh_down == 1
            testx = [squeeze(xGhost(i,:,end-1:end))'; testx];
            yy_cv_test = [yGhost(i,end-1:end)'; yy_cv_test];
        else
            testx = [xxi(tresh_down-2:tresh_down-1,:); testx];
            yy_cv_test = [yi(tresh_down-2:tresh_down-1); yy_cv_test];
        end
        
        trening= [squeeze(xGhost(i,:,:))'; xxi(1:(tresh_down-1),:)];
        ytre=[yGhost(i,:)'; yi(1:(tresh_down-1))];
        
        trening = trening(end-training_window+1:end,:);
        ytre = ytre(end-training_window+1:end);
        
        duzina = size(trening, 1);
        xx_lag0 = trening;
        yy_lag0 = ytre;
        xx_lag1 = [trening(1:duzina-1, :), trening(2:duzina, :), ytre(1:duzina-1)];
        yy_lag1 = ytre(2:end);
        xx_lag2 = [trening(1:duzina-2, :), trening(2:duzina-1, :), ...
            trening(3:duzina, :), ytre(1:duzina-2), ytre(2:duzina-1)];
        yy_lag2 = ytre(3:end);
        
        xx_lag0_cv_test = testx(3:end,:);
        xx_lag1_cv_test = [testx(2:end-1,:), testx(3:end,:), yy_cv_test(2:end-1)];
        xx_lag2_cv_test = [testx(1:end-2,:), testx(2:end-1,:), testx(3:end,:), yy_cv_test(1:end-2), yy_cv_test(2:end-1)];
        
        % Train Linear ARX
        mdl_lag0 = LinearModel.fit(xx_lag0, yy_lag0);
        mdl_lag1 = LinearModel.fit(xx_lag1, yy_lag1);
        mdl_lag2 = LinearModel.fit(xx_lag2, yy_lag2);
        
%         weights_lag0 = linearWeights(yy_lag0);
%         weights_lag1 = linearWeights(yy_lag1);
%         weights_lag2 = linearWeights(yy_lag2);
        
        % Train Lasso
        [B_lasso_lag0, FitInfo_lasso_lag0] = lasso(xx_lag0, yy_lag0, 'Alpha', 1, 'DFmax', 5, 'NumLambda', 100);
        [B_lasso_lag1, FitInfo_lasso_lag1] = lasso(xx_lag1, yy_lag1, 'Alpha', 1, 'DFmax', 7, 'NumLambda', 100);
        [B_lasso_lag2, FitInfo_lasso_lag2] = lasso(xx_lag2, yy_lag2, 'Alpha', 1, 'DFmax', 10, 'NumLambda', 100);
        
        % Train Elastic Net
        [B_elnet_lag0, FitInfo_elnet_lag0] = lasso(xx_lag0, yy_lag0,'Alpha',0.5, 'DFmax', 5,'NumLambda', 100);
        [B_elnet_lag1, FitInfo_elnet_lag1] = lasso(xx_lag1, yy_lag1,'Alpha',0.5, 'DFmax', 7,'NumLambda', 100);
        [B_elnet_lag2, FitInfo_elnet_lag2] = lasso(xx_lag2, yy_lag2,'Alpha',0.5, 'DFmax', 10,'NumLambda', 100);
        
        % Predict using Linear ARX
        R_linreg_lag0(i,tresh_down:tresh_up) = predict(mdl_lag0, xx_lag0_cv_test);
        R_linreg_lag1(i,tresh_down:tresh_up) = predict(mdl_lag1, xx_lag1_cv_test);
        R_linreg_lag2(i,tresh_down:tresh_up) = predict(mdl_lag2, xx_lag2_cv_test);
        
        % Predict using Lasso
        B_lasso_lag0 = selectLambdaLasso(B_lasso_lag0, FitInfo_lasso_lag0);
        R_lasso_lag0(i,tresh_down:tresh_up) = glmval(B_lasso_lag0, xx_lag0_cv_test, 'identity');
        B_lasso_lag1 = selectLambdaLasso(B_lasso_lag1, FitInfo_lasso_lag1);
        R_lasso_lag1(i,tresh_down:tresh_up) = glmval(B_lasso_lag1, xx_lag1_cv_test, 'identity');
        B_lasso_lag2 = selectLambdaLasso(B_lasso_lag2, FitInfo_lasso_lag2);
        R_lasso_lag2(i,tresh_down:tresh_up) = glmval(B_lasso_lag2, xx_lag2_cv_test, 'identity');
        
        % Predict using Elastic Net
        B_elnet_lag0 = selectLambdaLasso(B_elnet_lag0, FitInfo_elnet_lag0);
        R_elasticnet_lag0(i,tresh_down:tresh_up) = glmval(B_elnet_lag0, xx_lag0_cv_test, 'identity');
        B_elnet_lag1 = selectLambdaLasso(B_elnet_lag1, FitInfo_elnet_lag1);
        R_elasticnet_lag1(i,tresh_down:tresh_up) = glmval(B_elnet_lag1, xx_lag1_cv_test, 'identity');
        B_elnet_lag2 = selectLambdaLasso(B_elnet_lag2, FitInfo_elnet_lag2);
        R_elasticnet_lag2(i,tresh_down:tresh_up) = glmval(B_elnet_lag2, xx_lag2_cv_test, 'identity');
        
        % Calculate variance of Linear ARX
        yy_cv_true = yy_cv_test(3:end)';
        Var_linreg_lag0(i, tresh_down:tresh_up) = ...
            (yy_cv_true - R_linreg_lag0(i, tresh_down:tresh_up)).^2 / ...
            (size(xx_lag0_cv_test, 1) - size(xx_lag0_cv_test, 2) - 1);
        Var_linreg_lag1(i, tresh_down:tresh_up) = ...
            (yy_cv_true - R_linreg_lag1(i, tresh_down:tresh_up)).^2 / ...
            (size(xx_lag1_cv_test, 1) - size(xx_lag1_cv_test, 2) - 1);
        Var_linreg_lag2(i, tresh_down:tresh_up) = ...
            (yy_cv_true - R_linreg_lag2(i, tresh_down:tresh_up)).^2 / ...
            (size(xx_lag2_cv_test, 1) - size(xx_lag2_cv_test, 2) - 1);
        
        % Calculate variance of Lasso
        Var_lasso_lag0(i, tresh_down:tresh_up) = ...
            (yy_cv_true - R_lasso_lag0(i, tresh_down:tresh_up)).^2 / ...
            (size(xx_lag0_cv_test, 1) - size(xx_lag0_cv_test, 2) - 1);
        Var_lasso_lag1(i, tresh_down:tresh_up) = ...
            (yy_cv_true - R_lasso_lag1(i, tresh_down:tresh_up)).^2 / ...
            (size(xx_lag1_cv_test, 1) - size(xx_lag1_cv_test, 2) - 1);
        Var_lasso_lag2(i, tresh_down:tresh_up) = ...
            (yy_cv_true - R_lasso_lag2(i, tresh_down:tresh_up)).^2 / ...
            (size(xx_lag2_cv_test, 1) - size(xx_lag2_cv_test, 2) - 1);
        
        % Calculate variance of Elastic Net
        Var_elasticnet_lag0(i, tresh_down:tresh_up) = ...
            (yy_cv_true - R_elasticnet_lag0(i, tresh_down:tresh_up)).^2 / ...
            (size(xx_lag0_cv_test, 1) - size(xx_lag0_cv_test, 2) - 1);
        Var_elasticnet_lag1(i, tresh_down:tresh_up) = ...
            (yy_cv_true - R_elasticnet_lag1(i, tresh_down:tresh_up)).^2 / ...
            (size(xx_lag1_cv_test, 1) - size(xx_lag1_cv_test, 2) - 1);
        Var_elasticnet_lag2(i, tresh_down:tresh_up) = ...
            (yy_cv_true - R_elasticnet_lag2(i, tresh_down:tresh_up)).^2 / ...
            (size(xx_lag2_cv_test, 1) - size(xx_lag2_cv_test, 2) - 1);
        
        % Traing Gaussian Processes, Predict and calculate variance

        
        hyp0 = minimize(hyp0, @gp, -100, @infExact, meanfunc, covfunc, likfunc, xx_lag0, yy_lag0);
        hyp1 = minimize(hyp1, @gp, -100, @infExact, meanfunc, covfunc, likfunc, xx_lag1, yy_lag1);
        hyp2 = minimize(hyp2, @gp, -100, @infExact, meanfunc, covfunc, likfunc, xx_lag2, yy_lag2);
        
        [R_gp_lag0(i,tresh_down:tresh_up), Var_gp_lag0(i,tresh_down:tresh_up)] = ...
            gp(hyp0, @infExact, meanfunc, covfunc, likfunc, xx_lag0, yy_lag0, xx_lag0_cv_test);
        [R_gp_lag1(i,tresh_down:tresh_up), Var_gp_lag1(i,tresh_down:tresh_up)] = ...
            gp(hyp1, @infExact, meanfunc, covfunc, likfunc, xx_lag1, yy_lag1, xx_lag1_cv_test);
        [R_gp_lag2(i,tresh_down:tresh_up), Var_gp_lag2(i,tresh_down:tresh_up)] = ...
            gp(hyp2, @infExact, meanfunc, covfunc, likfunc, xx_lag2, yy_lag2, xx_lag2_cv_test);
        
    end
    
    trening= xxi(:,:);
    ytre= yi(:);
    ytre = ytre(end-training_window+1:end);
    trening = trening(end-training_window+1:end,:);
    
    ytest =  y(i,:)';
    ytest = ytest(T(2)-1:end);
    
    duzina = size(trening, 1);
    xx_lag0 = trening;
    yy_lag0 = ytre;
    xx_lag1 = [trening(1:duzina-1, :), trening(2:duzina, :), ytre(1:duzina-1)];
    yy_lag1 = ytre(2:end);
    xx_lag2 = [trening(1:duzina-2, :), trening(2:duzina-1, :), ...
        trening(3:duzina, :), ytre(1:duzina-2), ytre(2:duzina-1)];
    yy_lag2 = ytre(3:end);
    
    xx_lag0 = zscore(xx_lag0,[], 2);
    xx_lag1 = zscore(xx_lag1,[], 2);
    xx_lag2 = zscore(xx_lag2,[], 2);
    
    xx_lag0_test = xxi_test(3:end,:);
    xx_lag1_test = [xxi_test(2:end-1,:), xxi_test(3:end,:), ytest(2:end-1)];
    xx_lag2_test = [xxi_test(1:end-2,:), xxi_test(2:end-1,:), xxi_test(3:end,:), ytest(1:end-2), ytest(2:end-1)];
        
    xx_lag0_test = zscore(xx_lag0_test,[], 2);
    xx_lag1_test = zscore(xx_lag1_test,[], 2);
    xx_lag2_test = zscore(xx_lag2_test,[], 2);
    
    
    % Train Linear ARX
    mdl_lag0 = LinearModel.fit(xx_lag0, yy_lag0);
    mdl_lag1 = LinearModel.fit(xx_lag1, yy_lag1);
    mdl_lag2 = LinearModel.fit(xx_lag2, yy_lag2);
    
%     weights_lag0 = linearWeights(yy_lag0);
%     weights_lag1 = linearWeights(yy_lag1);
%     weights_lag2 = linearWeights(yy_lag2);
    
    % Train Linear ARX
    [B_lasso_lag0, FitInfo_lasso_lag0] = lasso(xx_lag0, yy_lag0, 'Alpha', 1, 'DFmax', 5, 'NumLambda', 100);
    [B_lasso_lag1, FitInfo_lasso_lag1] = lasso(xx_lag1, yy_lag1, 'Alpha', 1,'DFmax', 7, 'NumLambda', 100);
    [B_lasso_lag2, FitInfo_lasso_lag2] = lasso(xx_lag2, yy_lag2, 'Alpha', 1,'DFmax', 10, 'NumLambda', 100);
    
    % Train Linear ARX
    [B_elnet_lag0, FitInfo_elnet_lag0] = lasso(xx_lag0, yy_lag0, 'Alpha', 0.5, 'DFmax', 5, 'NumLambda', 100);
    [B_elnet_lag1, FitInfo_elnet_lag1] = lasso(xx_lag1, yy_lag1, 'Alpha', 0.5, 'DFmax', 7,'NumLambda', 100);
    [B_elnet_lag2, FitInfo_elnet_lag2] = lasso(xx_lag2, yy_lag2, 'Alpha', 0.5, 'DFmax', 10,'NumLambda', 100);    
    
    % Predictive weights for Lasso
    B_lasso_lag0 = selectLambdaLasso(B_lasso_lag0, FitInfo_lasso_lag0);
    B_lasso_lag1 = selectLambdaLasso(B_lasso_lag1, FitInfo_lasso_lag1);
    B_lasso_lag2 = selectLambdaLasso(B_lasso_lag2, FitInfo_lasso_lag2);
    
    % Predictive weights for Elastic Net
    B_elnet_lag0 = selectLambdaLasso(B_elnet_lag0, FitInfo_elnet_lag0);
    B_elnet_lag1 = selectLambdaLasso(B_elnet_lag1, FitInfo_elnet_lag1);
    B_elnet_lag2 = selectLambdaLasso(B_elnet_lag2, FitInfo_elnet_lag2);
    
    weights_linreg_lag0(i, :) = mdl_lag0.Coefficients.Estimate';
    weights_linreg_lag1(i, :) = mdl_lag1.Coefficients.Estimate';
    weights_linreg_lag2(i, :) = mdl_lag2.Coefficients.Estimate';
    weights_lasso_lag0(i, :) = B_lasso_lag0';
    weights_lasso_lag1(i, :) = B_lasso_lag1';
    weights_lasso_lag2(i, :) = B_lasso_lag2';
    weights_elasticnet_lag0(i, :) = B_elnet_lag0';
    weights_elasticnet_lag1(i, :) = B_elnet_lag1';
    weights_elasticnet_lag2(i, :) = B_elnet_lag2';

    %Fix uncertainty estimation now after CV is done on a node
    for j = T(2)+1:numTimeSteps
                
        xxx_lag0 = xx_lag0_test(j-T(2),:);
        xxx_lag1 = xx_lag1_test(j-T(2),:);
        xxx_lag2 = xx_lag2_test(j-T(2),:);
        
        % Predict using Linear ARX
        R_linreg_lag0(i,j) = predict(mdl_lag0, xxx_lag0);
        R_linreg_lag1(i,j) = predict(mdl_lag1, xxx_lag1);
        R_linreg_lag2(i,j) = predict(mdl_lag2, xxx_lag2);
        
        % Predict using Lasso
        R_lasso_lag0(i,j) = glmval(B_lasso_lag0, xxx_lag0, 'identity');
        R_lasso_lag1(i,j) = glmval(B_lasso_lag1, xxx_lag1, 'identity');
        R_lasso_lag2(i,j) = glmval(B_lasso_lag2, xxx_lag2, 'identity');
        
        % Predict using Elastic Net
        R_elasticnet_lag0(i,j) = glmval(B_elnet_lag0, xxx_lag0, 'identity');
        R_elasticnet_lag1(i,j) = glmval(B_elnet_lag1, xxx_lag1, 'identity');
        R_elasticnet_lag2(i,j) = glmval(B_elnet_lag2, xxx_lag2, 'identity');
        
        xxx_training_lag0 = squeeze(x(i,:,1:j-1))';
        xxx_training_lag1 = [squeeze(x(i,:,1:j-2))', squeeze(x(i,:,2:j-1))', y(i,1:j-2)'];
        xxx_training_lag2 = [squeeze(x(i,:,1:j-3))', squeeze(x(i,:,2:j-2))', squeeze(x(i,:,3:j-1))', y(i,1:j-3)', y(i,2:j-2)'];
        
        Var_linreg_lag0(i,j) = ...
            Var_linreg_lag0(i, j-1) .* ...
            diag(1 + xxx_lag0/(xxx_training_lag0' * xxx_training_lag0) *...
            xxx_lag0');
        Var_linreg_lag1(i,j) = ...
            Var_linreg_lag1(i, j-1) .* ...
            diag(1 + xxx_lag1/(xxx_training_lag1' * xxx_training_lag1) *...
            xxx_lag1');
        Var_linreg_lag2(i,j) = ...
            Var_linreg_lag2(i, j-1) .* ...
            diag(1 + xxx_lag2/(xxx_training_lag2' * xxx_training_lag2) *...
            xxx_lag2');
        
        Var_lasso_lag0(i,j) = ...
            Var_lasso_lag0(i, j-1) .* ...
            diag(1 + xxx_lag0/(xxx_training_lag0' * xxx_training_lag0) *...
            xxx_lag0');
        Var_lasso_lag1(i,j) = ...
            Var_lasso_lag1(i, j-1) .* ...
            diag(1 + xxx_lag1/(xxx_training_lag1' * xxx_training_lag1) *...
            xxx_lag1');
        Var_lasso_lag2(i,j) = ...
            Var_lasso_lag2(i, j-1) .* ...
            diag(1 + xxx_lag2/(xxx_training_lag2' * xxx_training_lag2) *...
            xxx_lag2');
        
        Var_elasticnet_lag0(i,j) = ...
            Var_elasticnet_lag0(i, j-1) .* ...
            diag(1 + xxx_lag0/(xxx_training_lag0' * xxx_training_lag0) *...
            xxx_lag0');
        Var_elasticnet_lag1(i,j) = ...
            Var_elasticnet_lag1(i, j-1) .* ...
            diag(1 + xxx_lag1/(xxx_training_lag1' * xxx_training_lag1) *...
            xxx_lag1');
        Var_elasticnet_lag2(i,j) = ...
            Var_elasticnet_lag2(i, j-1) .* ...
            diag(1 + xxx_lag2/(xxx_training_lag2' * xxx_training_lag2) *...
            xxx_lag2');
        
    end
    
    
    % Traing Gaussian Processes, Predict and calculate variance
    
    hyp0 = minimize(hyp0, @gp, -100, @infExact, meanfunc, covfunc, likfunc, xx_lag0, yy_lag0);
    hyp1 = minimize(hyp1, @gp, -100, @infExact, meanfunc, covfunc, likfunc, xx_lag1, yy_lag1);
    hyp2 = minimize(hyp2, @gp, -100, @infExact, meanfunc, covfunc, likfunc, xx_lag2, yy_lag2);
    
    [R_gp_lag0(i,T(2)+1:end), Var_gp_lag0(i,T(2)+1:end)] = ...
        gp(hyp0, @infExact, meanfunc, covfunc, likfunc,...
        xx_lag0, yy_lag0, xx_lag0_test);
    
    [R_gp_lag1(i,T(2)+1:end), Var_gp_lag1(i,T(2)+1:end)] = ...
        gp(hyp1, @infExact, meanfunc, covfunc, likfunc,...
        xx_lag1, yy_lag1, xx_lag1_test);
    
    [R_gp_lag2(i,T(2)+1:end), Var_gp_lag2(i,T(2)+1:end)] = ...
        gp(hyp2, @infExact, meanfunc, covfunc, likfunc,...
        xx_lag2, yy_lag2, xx_lag2_test);
    
    
end

predictors{1} = R_linreg_lag0;
predictors{2} = R_linreg_lag1;
predictors{3} = R_linreg_lag2;
predictors{4} = R_lasso_lag0;
predictors{5} = R_lasso_lag1;
predictors{6} = R_lasso_lag2;
predictors{7} = R_elasticnet_lag0;
predictors{8} = R_elasticnet_lag1;
predictors{9} = R_elasticnet_lag2;
predictors{10} = R_gp_lag0;
predictors{11} = R_gp_lag1;
predictors{12} = R_gp_lag2;

Var_linreg_lag0(Var_linreg_lag0 < 1e-4) = 1e-4;
variances{1} = Var_linreg_lag0;

Var_linreg_lag1(Var_linreg_lag1 < 1e-4) = 1e-4;
variances{2} = Var_linreg_lag1;

Var_linreg_lag2(Var_linreg_lag2 < 1e-4) = 1e-4;
variances{3} = Var_linreg_lag2;

Var_lasso_lag0(Var_lasso_lag0 < 1e-4) = 1e-4;
variances{4} = Var_lasso_lag0;

Var_lasso_lag1(Var_lasso_lag1 < 1e-4) = 1e-4;
variances{5} = Var_lasso_lag1;

Var_lasso_lag2(Var_lasso_lag2 < 1e-4) = 1e-4;
variances{6} = Var_lasso_lag2;

Var_elasticnet_lag0(Var_elasticnet_lag0 < 1e-4) = 1e-4;
variances{7} = Var_elasticnet_lag0;

Var_elasticnet_lag1(Var_elasticnet_lag1 < 1e-4) = 1e-4;
variances{8} = Var_elasticnet_lag1;

Var_elasticnet_lag2(Var_elasticnet_lag2 < 1e-4) = 1e-4;
variances{9} = Var_elasticnet_lag2;

Var_gp_lag0(Var_gp_lag0 < 1e-4) = 1e-4;
variances{10} = Var_gp_lag0;

Var_gp_lag1(Var_gp_lag1 < 1e-4) = 1e-4;
variances{11} = Var_gp_lag1;

Var_gp_lag2(Var_gp_lag2 < 1e-4) = 1e-4;
variances{12} = Var_gp_lag2;

Weights{1} = weights_linreg_lag0;
Weights{2} = weights_linreg_lag1;
Weights{3} = weights_linreg_lag2;
Weights{4} = weights_lasso_lag0;
Weights{5} = weights_lasso_lag1;
Weights{6} = weights_lasso_lag2;
Weights{7} = weights_elasticnet_lag0;
Weights{8} = weights_elasticnet_lag1;
Weights{9} = weights_elasticnet_lag2;

%% Calculating confidence interval guesses
confidence_interval_guesses = zeros(1,length(predictors));

yint = y(:, T(1): T(2));
yint = yint(:);

for i = 1 : size(yint,1)
    for j = 1 : length(predictors)
        pred = predictors{j}(:, T(1): T(2));
        pred = pred(:);
        sigma = variances{j}(:, T(1): T(2));
        sigma = sigma(:);
        if and(yint(i) > (pred(i) - 1.96*sqrt(sigma(i))), yint(i) < (pred(i) + 1.96*sqrt(sigma(i))))
            confidence_interval_guesses(1,j) = confidence_interval_guesses(1,j) + 1;
        end
    end
    
end
confidence_interval_guesses(confidence_interval_guesses==0)=confidence_interval_guesses(confidence_interval_guesses==0)+1; %ako bas nema ni jedan da kazemo da je bar jedan upao

confidence_guesses = confidence_interval_guesses./(N*(T(2)-T(1)+1));
printmat(confidence_guesses, 'Confidence interval guesses', 'percentage', 'arx_lag0 arx_lag1 arx_lag2 lasso_lag0 lasso_lag1 lasso_lag2 elnet_lag0 elnet_lag1 elnet_lag2 gp_lag0 gp_lag1 gp_lag2' )


end



