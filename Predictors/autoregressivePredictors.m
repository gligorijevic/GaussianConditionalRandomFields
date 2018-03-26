function [ predictors, variances, confidenceGuesses ] = autoregressivePredictors( y, x, yGhost, numTimeSteps, N, T, trainingWindow, lagSize )

net_lag0 = fitnet(5);
net_lag0.trainParam.show = NaN;
net_lag0.trainParam.showWindow = false;
net_lag1 = fitnet(7);
net_lag1.trainParam.show = NaN;
net_lag1.trainParam.showWindow = false;
net_lag2 = fitnet(9);
net_lag2.trainParam.show = NaN;
net_lag2.trainParam.showWindow = false;

addpath(genpath('..\gpml-matlab-v3.4-2013-11-11\'));

% meanfunc = {@meanSum, {@meanLinear, @meanConst}};
meanfunc = {@meanZero};

M = 0;

covfunc = {@covSEard};
L = ones(M+1, 1); sf = 1; hyp0.cov = log([L; sf]);
L = ones(M*2+2, 1); sf = 1; hyp1.cov = log([L; sf]); %adding one for y from previous timestep
L = ones(M*3+3, 1); sf = 1; hyp2.cov = log([L; sf]); %adding two for y from previous timesteps
likfunc = @likGauss; sn = 5; hyp0.lik = log(sn); hyp1.lik = log(sn); hyp2.lik = log(sn);
inffunc = @infExact;

R_linreg_lag0 = zeros(N, numTimeSteps);
R_linreg_lag1 = zeros(N, numTimeSteps);
R_linreg_lag2 = zeros(N, numTimeSteps);
Var_linreg_lag0 = zeros(N, numTimeSteps);
Var_linreg_lag1 = zeros(N, numTimeSteps);
Var_linreg_lag2 = zeros(N, numTimeSteps);

% % % % R_lasso_lag0 = zeros(N, numTimeSteps);
% % % % R_lasso_lag1 = zeros(N, numTimeSteps);
% % % % R_lasso_lag2 = zeros(N, numTimeSteps);
% % % % Var_lasso_lag0 = zeros(N, numTimeSteps);
% % % % Var_lasso_lag1 = zeros(N, numTimeSteps);
% % % % Var_lasso_lag2 = zeros(N, numTimeSteps);

% % % % R_elasticnet_lag0 = zeros(N, numTimeSteps);
% % % % R_elasticnet_lag1 = zeros(N, numTimeSteps);
% % % % R_elasticnet_lag2 = zeros(N, numTimeSteps);
% % % % Var_elasticnet_lag0 = zeros(N, numTimeSteps);
% % % % Var_elasticnet_lag1 = zeros(N, numTimeSteps);
% % % % Var_elasticnet_lag2 = zeros(N, numTimeSteps);

R_gp_lag0 = zeros(N, numTimeSteps);
R_gp_lag1 = zeros(N, numTimeSteps);
R_gp_lag2 = zeros(N, numTimeSteps);
Var_gp_lag0 = zeros(N, numTimeSteps);
Var_gp_lag1 = zeros(N, numTimeSteps);
Var_gp_lag2 = zeros(N, numTimeSteps);

% R_nn_lag0 = zeros(N, numTimeSteps);
% R_nn_lag1 = zeros(N, numTimeSteps);
% R_nn_lag2 = zeros(N, numTimeSteps);

% weights_linreg_lag0 = zeros(N, M+1);
% weights_linreg_lag1 = zeros(N, M*2+1+1);
% weights_linreg_lag2 = zeros(N, M*3+2+1);
% weights_lasso_lag0 = zeros(N, M+1);
% weights_lasso_lag1 = zeros(N, M*2+1+1);
% weights_lasso_lag2 = zeros(N, M*3+2+1);
% weights_elasticnet_lag0 = zeros(N, M+1);
% weights_elasticnet_lag1 = zeros(N, M*2+1+1);
% weights_elasticnet_lag2 = zeros(N, M*3+2+1);


%% K-fold Cross Validation

for i = 1:N
    %     fold = 4;
    
    %     xi = x(i,:,1:end);
    %     xi_test=x(i,:, T(2)+1:end);
    
    %     xxi = reshape(ipermute(xi,[1 3 2]),[],size(x,2));
    %     xxi_test=reshape(ipermute(xi_test,[1 3 2]),[],size(x,2));
    
    %     xxGhost = xGhost(i,:,:);
    %     xxGhost = reshape(ipermute(xxGhost,[1 3 2]),[],size(xxGhost,2));
    
    %     yi=y(i,T(1):T(2))';
    yi_test = y(i,1:end);
    for iterval=1:numTimeSteps
        
        %         testx = xxi;
        yy_cv_test = yi_test(iterval);
        
        if iterval == 1
            %             parce_x = [];
            parce_y = [];
        else
            %             parce_x = [xxi(1:iterval-1,:)];
            parce_y = [yi_test(1:iterval-1)];
        end
        
        %         trening= [xxGhost; parce_x];
        ytre=[yGhost(i,:)'; parce_y'];
        
        %         trening = trening(end-trainingWindow+1:end,:);
        ytre = ytre(end-trainingWindow+1:end);
        
        duzina = size(ytre, 1);
        xx_lag0 = [ytre(1:duzina-1)];
        yy_lag0 = ytre(2:end);
        xx_lag1 = [ytre(1:duzina-2), ytre(2:duzina-1)];
        yy_lag1 = ytre(3:end);
        xx_lag2 = [ytre(1:duzina-3), ytre(2:duzina-2), ytre(3:duzina-1)];
        yy_lag2 = ytre(4:end);
        
        if iterval > 3
            xx_lag0_cv_test = [yi_test(1,iterval-1)];
            xx_lag1_cv_test = [yi_test(1,iterval-2), yi_test(1,iterval-1)];
            xx_lag2_cv_test = [yi_test(1,iterval-3), yi_test(1,iterval-2), yi_test(1,iterval-1)];
        else if iterval == 3
                %                 xx_lag0_cv_test = testx(iterval,:);
                %                 xx_lag1_cv_test = [testx(iterval,:), testx(iterval-1,:), yi_test(iterval-1)];
                %                 xx_lag2_cv_test = [testx(iterval,:), testx(iterval-1,:), xxGhost(end,:), yGhost(i,end), yi_test(iterval-1)];
                xx_lag0_cv_test = [yi_test(1,iterval-1)];
                xx_lag1_cv_test = [yi_test(1,iterval-2), yi_test(1,iterval-1)];
                xx_lag2_cv_test = [yGhost(i,end), yi_test(1,iterval-2), yi_test(1,iterval-1)];
            else if iterval == 2
                    xx_lag0_cv_test = [yi_test(1,iterval-1,:)];
                    xx_lag1_cv_test = [yGhost(i,end), yi_test(1,iterval-1,:)];
                    xx_lag2_cv_test = [yGhost(i,end-1), yGhost(i,end), yi_test(1,iterval-1,:)];
                else if iterval == 1
                        xx_lag0_cv_test = [yGhost(i,end)];
                        xx_lag1_cv_test = [yGhost(i,end-1), yGhost(i,end)];
                        xx_lag2_cv_test = [yGhost(i,end-2), yGhost(i,end-1), yGhost(i,end)];
                    end
                end
            end
        end
        
        if lagSize == 3
            
            % Train Linear ARX
            mdl_lag0 = LinearModel.fit(xx_lag0, yy_lag0);
            mdl_lag1 = LinearModel.fit(xx_lag1, yy_lag1);
            mdl_lag2 = LinearModel.fit(xx_lag2, yy_lag2);
            
            %         weights_lag0 = linearWeights(yy_lag0);
            %         weights_lag1 = linearWeights(yy_lag1);
            %         weights_lag2 = linearWeights(yy_lag2);
            %
            % Train Lasso
            % % % %             [B_lasso_lag0, FitInfo_lasso_lag0] = lasso(xx_lag0, yy_lag0, 'Alpha', 1, 'DFmax', 5, 'NumLambda', 100);
            % % % %             [B_lasso_lag1, FitInfo_lasso_lag1] = lasso(xx_lag1, yy_lag1, 'Alpha', 1, 'DFmax', 7, 'NumLambda', 100);
            % % % %             [B_lasso_lag2, FitInfo_lasso_lag2] = lasso(xx_lag2, yy_lag2, 'Alpha', 1, 'DFmax', 10, 'NumLambda', 100);
            
            % Train Elastic Net
            % % % %             [B_elnet_lag0, FitInfo_elnet_lag0] = lasso(xx_lag0, yy_lag0,'Alpha',0.5, 'DFmax', 5,'NumLambda', 100);
            % % % %             [B_elnet_lag1, FitInfo_elnet_lag1] = lasso(xx_lag1, yy_lag1,'Alpha',0.5, 'DFmax', 7,'NumLambda', 100);
            % % % %             [B_elnet_lag2, FitInfo_elnet_lag2] = lasso(xx_lag2, yy_lag2,'Alpha',0.5, 'DFmax', 10,'NumLambda', 100);
            
            % Predict using Linear ARX
            R_linreg_lag0(i,iterval) = predict(mdl_lag0, xx_lag0_cv_test);
            R_linreg_lag1(i,iterval) = predict(mdl_lag1, xx_lag1_cv_test);
            R_linreg_lag2(i,iterval) = predict(mdl_lag2, xx_lag2_cv_test);
            
            % Predict using Lasso
            % % % %             B_lasso_lag0 = selectLambdaLasso(B_lasso_lag0, FitInfo_lasso_lag0);
            % % % %             R_lasso_lag0(i,iterval) = glmval(B_lasso_lag0, xx_lag0_cv_test, 'identity');
            % % % %             B_lasso_lag1 = selectLambdaLasso(B_lasso_lag1, FitInfo_lasso_lag1);
            % % % %             R_lasso_lag1(i,iterval) = glmval(B_lasso_lag1, xx_lag1_cv_test, 'identity');
            % % % %             B_lasso_lag2 = selectLambdaLasso(B_lasso_lag2, FitInfo_lasso_lag2);
            % % % %             R_lasso_lag2(i,iterval) = glmval(B_lasso_lag2, xx_lag2_cv_test, 'identity');
            
            % Predict using Elastic Net
            % % % %             B_elnet_lag0 = selectLambdaLasso(B_elnet_lag0, FitInfo_elnet_lag0);
            % % % %             R_elasticnet_lag0(i,iterval) = glmval(B_elnet_lag0, xx_lag0_cv_test, 'identity');
            % % % %             B_elnet_lag1 = selectLambdaLasso(B_elnet_lag1, FitInfo_elnet_lag1);
            % % % %             R_elasticnet_lag1(i,iterval) = glmval(B_elnet_lag1, xx_lag1_cv_test, 'identity');
            % % % %             B_elnet_lag2 = selectLambdaLasso(B_elnet_lag2, FitInfo_elnet_lag2);
            % % % %             R_elasticnet_lag2(i,iterval) = glmval(B_elnet_lag2, xx_lag2_cv_test, 'identity');
            
            % Calculate variance of Linear ARX
            % % % %             yy_cv_true = yi_test(iterval)';
            % % % %             Var_linreg_lag0(i, iterval) = ...
            % % % %                 (yy_cv_true - R_linreg_lag0(i, iterval)).^2 / ...
            % % % %                 (size(xx_lag0_cv_test, 1) - size(xx_lag0_cv_test, 2));
            % % % %             Var_linreg_lag1(i, iterval) = ...
            % % % %                 (yy_cv_true - R_linreg_lag1(i, iterval)).^2 / ...
            % % % %                 (size(xx_lag1_cv_test, 1));
            % % % %             Var_linreg_lag2(i, iterval) = ...
            % % % %                 (yy_cv_true - R_linreg_lag2(i, iterval)).^2 / ...
            % % % %                 (size(xx_lag2_cv_test, 1);
            
            %kasapin
            AA = sum((yy_lag0 - predict(mdl_lag0, xx_lag0)).^2) / (size(xx_lag0, 1) - size(xx_lag0, 2));
            BB = sum((yy_lag1 - predict(mdl_lag1, xx_lag1)).^2) / (size(xx_lag1, 1) - size(xx_lag1, 2));
            CC = sum((yy_lag2 - predict(mdl_lag2, xx_lag2)).^2) / (size(xx_lag2, 1) - size(xx_lag2, 2));
            
            Var_linreg_lag0(i,iterval) = AA .* diag(1 + xx_lag0_cv_test/(xx_lag0' * xx_lag0) * xx_lag0_cv_test');
            Var_linreg_lag1(i,iterval) = BB .* diag(1 + xx_lag1_cv_test/(xx_lag1' * xx_lag1) * xx_lag1_cv_test');
            Var_linreg_lag2(i,iterval) = CC .* diag(1 + xx_lag2_cv_test/(xx_lag2' * xx_lag2) * xx_lag2_cv_test');
            
            % Calculate variance of Lasso
            % % % %             Var_lasso_lag0(i, iterval) = ...
            % % % %                 (yy_cv_true - R_lasso_lag0(i, iterval)).^2 / ...
            % % % %                 (size(xx_lag0_cv_test, 1));
            % % % %             Var_lasso_lag1(i, iterval) = ...
            % % % %                 (yy_cv_true - R_lasso_lag1(i, iterval)).^2 / ...
            % % % %                 (size(xx_lag1_cv_test, 1));
            % % % %             Var_lasso_lag2(i, iterval) = ...
            % % % %                 (yy_cv_true - R_lasso_lag2(i, iterval)).^2 / ...
            % % % %                 (size(xx_lag2_cv_test, 1));
            
            % Calculate variance of Elastic Net
            % % % %             Var_elasticnet_lag0(i, iterval) = ...
            % % % %                 (yy_cv_true - R_elasticnet_lag0(i, iterval)).^2 / ...
            % % % %                 (size(xx_lag0_cv_test, 1) - 1);
            % % % %             Var_elasticnet_lag1(i, iterval) = ...
            % % % %                 (yy_cv_true - R_elasticnet_lag1(i, iterval)).^2 / ...
            % % % %                 (size(xx_lag1_cv_test, 1) - 1);
            % % % %             Var_elasticnet_lag2(i, iterval) = ...
            % % % %                 (yy_cv_true - R_elasticnet_lag2(i, iterval)).^2 / ...
            % % % %                 (size(xx_lag2_cv_test, 1) - 1);
            
            %Traing Gaussian Processes, Predict and calculate variance
            hyp0 = minimize(hyp0, @gp, -100, inffunc, meanfunc, covfunc, likfunc, xx_lag0, yy_lag0);
            hyp1 = minimize(hyp1, @gp, -100, inffunc, meanfunc, covfunc, likfunc, xx_lag1, yy_lag1);
            hyp2 = minimize(hyp2, @gp, -100, inffunc, meanfunc, covfunc, likfunc, xx_lag2, yy_lag2);
            
            [R_gp_lag0(i,iterval), Var_gp_lag0(i,iterval)] = ...
                gp(hyp0, inffunc, meanfunc, covfunc, likfunc, xx_lag0, yy_lag0, xx_lag0_cv_test);
            [R_gp_lag1(i,iterval), Var_gp_lag1(i,iterval)] = ...
                gp(hyp1, inffunc, meanfunc, covfunc, likfunc, xx_lag1, yy_lag1, xx_lag1_cv_test);
            [R_gp_lag2(i,iterval), Var_gp_lag2(i,iterval)] = ...
                gp(hyp2, inffunc, meanfunc, covfunc, likfunc, xx_lag2, yy_lag2, xx_lag2_cv_test);
            
            %Training Neural Networks
            %             [net_lag0,tr] = train(net_lag0,xx_lag0',yy_lag0');
            %             R_nn_lag0(i,iterval) = net_lag0(xx_lag0_cv_test');
            %
            %             [net_lag1,tr] = train(net_lag1,xx_lag1',yy_lag1');
            %             R_nn_lag1(i,iterval) = net_lag1(xx_lag1_cv_test');
            %
            %             [net_lag2, tr] = train(net_lag2,xx_lag2',yy_lag2');
            %             R_nn_lag2(i,iterval) = net_lag2(xx_lag2_cv_test');
        else if lagSize == 2
                % Train Linear ARX
                mdl_lag0 = LinearModel.fit(xx_lag0, yy_lag0);
                mdl_lag1 = LinearModel.fit(xx_lag1, yy_lag1);
                
                % Train Lasso
                % % % %                 [B_lasso_lag0, FitInfo_lasso_lag0] = lasso(xx_lag0, yy_lag0, 'Alpha', 1, 'DFmax', 5, 'NumLambda', 100);
                % % % %                 [B_lasso_lag1, FitInfo_lasso_lag1] = lasso(xx_lag1, yy_lag1, 'Alpha', 1, 'DFmax', 7, 'NumLambda', 100);
                
                % Train Elastic Net
                % % % %                 [B_elnet_lag0, FitInfo_elnet_lag0] = lasso(xx_lag0, yy_lag0,'Alpha',0.5, 'DFmax', 5,'NumLambda', 100);
                % % % %                 [B_elnet_lag1, FitInfo_elnet_lag1] = lasso(xx_lag1, yy_lag1,'Alpha',0.5, 'DFmax', 7,'NumLambda', 100);
                
                % Predict using Linear ARX
                R_linreg_lag0(i,iterval) = predict(mdl_lag0, xx_lag0_cv_test);
                R_linreg_lag1(i,iterval) = predict(mdl_lag1, xx_lag1_cv_test);
                
                % Predict using Lasso
                % % % %                 B_lasso_lag0 = selectLambdaLasso(B_lasso_lag0, FitInfo_lasso_lag0);
                % % % %                 R_lasso_lag0(i,iterval) = glmval(B_lasso_lag0, xx_lag0_cv_test, 'identity');
                % % % %                 B_lasso_lag1 = selectLambdaLasso(B_lasso_lag1, FitInfo_lasso_lag1);
                % % % %                 R_lasso_lag1(i,iterval) = glmval(B_lasso_lag1, xx_lag1_cv_test, 'identity');
                
                % Predict using Elastic Net
                % % % %                 B_elnet_lag0 = selectLambdaLasso(B_elnet_lag0, FitInfo_elnet_lag0);
                % % % %                 R_elasticnet_lag0(i,iterval) = glmval(B_elnet_lag0, xx_lag0_cv_test, 'identity');
                % % % %                 B_elnet_lag1 = selectLambdaLasso(B_elnet_lag1, FitInfo_elnet_lag1);
                % % % %                 R_elasticnet_lag1(i,iterval) = glmval(B_elnet_lag1, xx_lag1_cv_test, 'identity');
                
                % Calculate variance of Linear ARX
                % THIS PART IS NOT VALID
                % % % % %                 yy_cv_true = yi_test(iterval)';
                % % % % %                 Var_linreg_lag0(i, iterval) = ...
                % % % % %                     (yy_cv_true - R_linreg_lag0(i, iterval)).^2 / ...
                % % % % %                     (size(xx_lag0_cv_test, 1) - 1);
                % % % % %                 Var_linreg_lag1(i, iterval) = ...
                % % % % %                     (yy_cv_true - R_linreg_lag1(i, iterval)).^2 / ...
                % % % % %                     (size(xx_lag1_cv_test, 1) - 1);
                AA = sum((yy_lag0 - predict(mdl_lag0, xx_lag0)).^2) / (size(xx_lag0, 1) - size(xx_lag0, 2));
                BB = sum((yy_lag2 - predict(mdl_lag1, xx_lag1)).^2) / (size(xx_lag1, 1) - size(xx_lag1, 2));
                
                Var_linreg_lag0(i,iterval) = AA .* diag(1 + xx_lag0_cv_test/(xx_lag0' * xx_lag0) * xx_lag0_cv_test');
                Var_linreg_lag1(i,iterval) = BB .* diag(1 + xx_lag1_cv_test/(xx_lag1' * xx_lag1) * xx_lag1_cv_test');
                
                % Calculate variance of Lasso
                % % % %                 Var_lasso_lag0(i, iterval) = ...
                % % % %                     (yy_cv_true - R_lasso_lag0(i, iterval)).^2 / ...
                % % % %                     (size(xx_lag0_cv_test, 1) - 1);
                % % % %                 Var_lasso_lag1(i, iterval) = ...
                % % % %                     (yy_cv_true - R_lasso_lag1(i, iterval)).^2 / ...
                % % % %                     (size(xx_lag1_cv_test, 1) - 1);
                
                % Calculate variance of Elastic Net
                % % % %                 Var_elasticnet_lag0(i, iterval) = ...
                % % % %                     (yy_cv_true - R_elasticnet_lag0(i, iterval)).^2 / ...
                % % % %                     (size(xx_lag0_cv_test, 1) - 1);
                % % % %                 Var_elasticnet_lag1(i, iterval) = ...
                % % % %                     (yy_cv_true - R_elasticnet_lag1(i, iterval)).^2 / ...
                % % % %                     (size(xx_lag1_cv_test, 1) - 1);
                
                %Traing Gaussian Processes, Predict and calculate variance
                hyp0 = minimize(hyp0, @gp, -100, inffunc, meanfunc, covfunc, likfunc, xx_lag0, yy_lag0);
                hyp1 = minimize(hyp1, @gp, -100, inffunc, meanfunc, covfunc, likfunc, xx_lag1, yy_lag1);
                
                [R_gp_lag0(i,iterval), Var_gp_lag0(i,iterval)] = ...
                    gp(hyp0, inffunc, meanfunc, covfunc, likfunc, xx_lag0, yy_lag0, xx_lag0_cv_test);
                [R_gp_lag1(i,iterval), Var_gp_lag1(i,iterval)] = ...
                    gp(hyp1, inffunc, meanfunc, covfunc, likfunc, xx_lag1, yy_lag1, xx_lag1_cv_test);
                
                %Training Neural Networks
                %                 [net_lag0,tr] = train(net_lag0,xx_lag0',yy_lag0');
                %                 R_nn_lag0(i,iterval) = net_lag0(xx_lag0_cv_test');
                %
                %                 [net_lag1,tr] = train(net_lag1,xx_lag1',yy_lag1');
                %                 R_nn_lag1(i,iterval) = net_lag1(xx_lag1_cv_test');
                
            else if lagSize == 1
                    % Train Linear ARX
                    mdl_lag0 = LinearModel.fit(xx_lag0, yy_lag0);
                    
                    % Train Lasso
                    % % % %                     [B_lasso_lag0, FitInfo_lasso_lag0] = lasso(xx_lag0, yy_lag0, 'Alpha', 1, 'DFmax', 5, 'NumLambda', 100);
                    
                    % Train Elastic Net
                    % % % %                     [B_elnet_lag0, FitInfo_elnet_lag0] = lasso(xx_lag0, yy_lag0,'Alpha',0.5, 'DFmax', 5,'NumLambda', 100);
                    
                    % Predict using Linear ARX
                    R_linreg_lag0(i,iterval) = predict(mdl_lag0, xx_lag0_cv_test);
                    
                    % Predict using Lasso
                    % % % %                     B_lasso_lag0 = selectLambdaLasso(B_lasso_lag0, FitInfo_lasso_lag0);
                    % % % %                     R_lasso_lag0(i,iterval) = glmval(B_lasso_lag0, xx_lag0_cv_test, 'identity');
                    
                    % Predict using Elastic Net
                    % % % %                     B_elnet_lag0 = selectLambdaLasso(B_elnet_lag0, FitInfo_elnet_lag0);
                    % % % %                     R_elasticnet_lag0(i,iterval) = glmval(B_elnet_lag0, xx_lag0_cv_test, 'identity');
                    
                    % Calculate variance of Linear ARX
                    %THIS PART IS NOT VALID
                    % % % %                     yy_cv_true = yi_test(iterval)';
                    % % % %                     Var_linreg_lag0(i, iterval) = ...
                    % % % %                         (yy_cv_true - R_linreg_lag0(i, iterval)).^2 / ...
                    % % % %                         (size(xx_lag0_cv_test, 1) - size(xx_lag0_cv_test, 2) - 1);
                    
                    AA = sum((yy_lag0 - predict(mdl_lag0, xx_lag0)).^2) / (size(xx_lag0, 1) - size(xx_lag0, 2));
                    
                    Var_linreg_lag0(i,iterval) = AA .* diag(1 + xx_lag0_cv_test/(xx_lag0' * xx_lag0) * xx_lag0_cv_test');
                    
                    % Calculate variance of Lasso
                    % % % %                     Var_lasso_lag0(i, iterval) = ...
                    % % % %                         (yy_cv_true - R_lasso_lag0(i, iterval)).^2 / ...
                    % % % %                         (size(xx_lag0_cv_test, 1) - 1);
                    
                    % Calculate variance of Elastic Net
                    % % % %                     Var_elasticnet_lag0(i, iterval) = ...
                    % % % %                         (yy_cv_true - R_elasticnet_lag0(i, iterval)).^2 / ...
                    % % % %                         (size(xx_lag0_cv_test, 1) - 1);
                    
                    %Traing Gaussian Processes, Predict and calculate variance
                    hyp0 = minimize(hyp0, @gp, -100, inffunc, meanfunc, covfunc, likfunc, xx_lag0, yy_lag0);
                    
                    [R_gp_lag0(i,iterval), Var_gp_lag0(i,iterval)] = ...
                        gp(hyp0, inffunc, meanfunc, covfunc, likfunc, xx_lag0, yy_lag0, xx_lag0_cv_test);
                    
                    %Training Neural Networks
                    %                     [net_lag0,tr] = train(net_lag0,xx_lag0',yy_lag0');
                    %                     R_nn_lag0(i,iterval) = net_lag0(xx_lag0_cv_test');
                    
                end
            end
        end
    end
    
end

if lagSize ==3
    predictors{1} = R_linreg_lag0;
    predictors{2} = R_linreg_lag1;
    predictors{3} = R_linreg_lag2;
    % % % %     predictors{4} = R_lasso_lag0;
    % % % %     predictors{5} = R_lasso_lag1;
    % % % %     predictors{6} = R_lasso_lag2;
    % % % %     predictors{7} = R_elasticnet_lag0;
    % % % %     predictors{8} = R_elasticnet_lag1;
    % % % %     predictors{9} = R_elasticnet_lag2;
    predictors{4} = R_gp_lag0;
    predictors{5} = R_gp_lag1;
    predictors{6} = R_gp_lag2;
    %     predictors{13} = R_nn_lag0;
    %     predictors{14} = R_nn_lag1;
    %     predictors{15} = R_nn_lag2;
    
    %     Var_linreg_lag0(Var_linreg_lag0 < 1e-8) = 1e-8;
    variances{1} = Var_linreg_lag0;
    
    %     Var_linreg_lag1(Var_linreg_lag1 < 1e-8) = 1e-8;
    variances{2} = Var_linreg_lag1;
    
    %     Var_linreg_lag2(Var_linreg_lag2 < 1e-8) = 1e-8;
    variances{3} = Var_linreg_lag2;
    
    %     Var_lasso_lag0(Var_lasso_lag0 < 1e-8) = 1e-8;
    % % % %     variances{4} = Var_lasso_lag0;
    
    %     Var_lasso_lag1(Var_lasso_lag1 < 1e-8) = 1e-8;
    % % % %     variances{5} = Var_lasso_lag1;
    
    %     Var_lasso_lag2(Var_lasso_lag2 < 1e-8) = 1e-8;
    % % % %     variances{6} = Var_lasso_lag2;
    
    %     Var_elasticnet_lag0(Var_elasticnet_lag0 < 1e-8) = 1e-8;
    % % % %     variances{7} = Var_elasticnet_lag0;
    
    %     Var_elasticnet_lag1(Var_elasticnet_lag1 < 1e-8) = 1e-8;
    % % % %     variances{8} = Var_elasticnet_lag1;
    
    %     Var_elasticnet_lag2(Var_elasticnet_lag2 < 1e-8) = 1e-8;
    % % % %     variances{9} = Var_elasticnet_lag2;
    
    %     Var_gp_lag0(Var_gp_lag0 < 1e-8) = 1e-8;
    variances{4} = Var_gp_lag0;
    
    %     Var_gp_lag1(Var_gp_lag1 < 1e-8) = 1e-8;
    variances{5} = Var_gp_lag1;
    
    %     Var_gp_lag2(Var_gp_lag2 < 1e-8) = 1e-8;
    variances{6} = Var_gp_lag2;
else if lagSize == 2
        
        predictors{1} = R_linreg_lag0;
        predictors{2} = R_linreg_lag1;
        % % % %         predictors{3} = R_lasso_lag0;
        % % % %         predictors{4} = R_lasso_lag1;
        % % % %         predictors{5} = R_elasticnet_lag0;
        % % % %         predictors{6} = R_elasticnet_lag1;
        predictors{3} = R_gp_lag0;
        predictors{4} = R_gp_lag1;
        %         predictors{9} = R_nn_lag0;
        %         predictors{10} = R_nn_lag1;
        
        Var_linreg_lag0(Var_linreg_lag0 < 1e-4) = 1e-4;
        variances{1} = Var_linreg_lag0;
        
        Var_linreg_lag1(Var_linreg_lag1 < 1e-4) = 1e-4;
        variances{2} = Var_linreg_lag1;
        
        % % % %         Var_lasso_lag0(Var_lasso_lag0 < 1e-4) = 1e-4;
        % % % %         variances{3} = Var_lasso_lag0;
        
        % % % %         Var_lasso_lag1(Var_lasso_lag1 < 1e-4) = 1e-4;
        % % % %         variances{4} = Var_lasso_lag1;
        
        % % % %         Var_elasticnet_lag0(Var_elasticnet_lag0 < 1e-4) = 1e-4;
        % % % %         variances{5} = Var_elasticnet_lag0;
        
        % % % %         Var_elasticnet_lag1(Var_elasticnet_lag1 < 1e-4) = 1e-4;
        % % % %         variances{6} = Var_elasticnet_lag1;
        
        Var_gp_lag0(Var_gp_lag0 < 1e-4) = 1e-4;
        variances{3} = Var_gp_lag0;
        
        Var_gp_lag1(Var_gp_lag1 < 1e-4) = 1e-4;
        variances{4} = Var_gp_lag1;
    else if lagSize ==1
            predictors{1} = R_linreg_lag0;
            % % % %             predictors{2} = R_lasso_lag0;
            % % % %             predictors{3} = R_elasticnet_lag0;
            predictors{2} = R_gp_lag0;
            %             predictors{5} = R_nn_lag0;
            
            Var_linreg_lag0(Var_linreg_lag0 < 1e-4) = 1e-4;
            variances{1} = Var_linreg_lag0;
            
            % % % %             Var_lasso_lag0(Var_lasso_lag0 < 1e-4) = 1e-4;
            % % % %             variances{2} = Var_lasso_lag0;
            
            % % % %             Var_elasticnet_lag0(Var_elasticnet_lag0 < 1e-4) = 1e-4;
            % % % %             variances{3} = Var_elasticnet_lag0;
            
            Var_gp_lag0(Var_gp_lag0 < 1e-4) = 1e-4;
            variances{2} = Var_gp_lag0;
        end
    end
end


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

%ako bas nema ni jedan da kazemo da je bar jedan upao
confidence_interval_guesses(confidence_interval_guesses==0) = ...
    confidence_interval_guesses(confidence_interval_guesses==0) + 1;

confidenceGuesses = confidence_interval_guesses./(N*(T(2)-T(1)+1));
printmat(confidenceGuesses, 'Confidence interval guesses')


end

