function Data = createXCRFstruct(x, ytr, ytest, predictors, similarities, Data, xunstr, xsim, nalpha, nbeta, nthetas_Rk, nthetas_Sl, lambdaAlpha, lambdaBeta, lambdaR, lambdaS, thetaAlpha, thetaBeta, thetaR, thetaS, lag, maxiter, useWorkers)
% This function creates Data object that contains data necessary for GCRF.

%% temporal settings
Data.Ttr = size(ytr,2); % function Data = precmatCRF(R,ytr, ytest,locStations, Data, nalpha, nbeta, spatialScale)
Data.Ttest = size(ytest,2);
[N T] = size([ytr ytest]);
Data.T = T;
Data.N = N;
%% Core data
Data.x = x;
Data.y = [ytr ytest];
Data.y_flat = Data.y(:);
Data.predictors = predictors;
Data.noUnstrPreds = length(predictors);
Data.noSimilarities = length(similarities);
Data.similarities = similarities;

%% Initial parameters values
Data.thetaR = thetaR;
Data.thetaS = thetaS;
Data.thetaAlpha = thetaAlpha;
Data.thetaBeta = thetaBeta;

%% Utility parameters
Data.lag = lag
Data.nalpha = nalpha;
Data.nbeta = nbeta; %Data.noSimilarities
Data.nthetas_Rk = nthetas_Rk;
Data.nthetas_Sl = nthetas_Sl;
Data.xunstr = xunstr;
Data.xsim = xsim;
Data.lambdaAlpha = lambdaAlpha;
Data.lambdaBeta = lambdaBeta;
Data.lambdaR = lambdaR;
Data.lambdaS = lambdaS;

%% Optimization parameters
Data.useWorkers = useWorkers; %true or false
Data.maxiter = maxiter;

%% Labeled data indexes
Data.label = [~isnan(ytr) zeros(size(ytest))];
Data.label = logical(Data.label(:));

%% Utility matrices for speedup
% M0 = sparse(diag(ones(1,T)));
% betaMatrix{1} = kron(M0, similarities{1}{1});

% alphaMatrix{1} = speye(N*T,N*T);
% M0 = sparse(diag(ones(1,T)));
% betaMatrix{1} = kron(M0, similarities{1}{1});
% Data.betaMatrix = betaMatrix;
% Data.alphaMatrix = alphaMatrix;

Data.X_unstr = {};
Data.X_sim = {};
[xx, yy] = meshgrid(1:N,1:N);

for nts = 1:T
    iks = Data.x(:, :, nts);
    X = reshape(ipermute(iks,[1 3 2]), [], size(iks,2));
    
    % Added bias term for linear regression
    Data.X_all{nts} = X;
    Data.X_unstr{nts} = [ones(size(X,1),1), X(:,Data.xunstr)];
    Data.X_sim{nts} = X(:,Data.xsim);

    Data.X_sim_dist_sq{nts} = (Data.X_sim{nts}(xx,:)-Data.X_sim{nts}(yy,:)).^2;
end