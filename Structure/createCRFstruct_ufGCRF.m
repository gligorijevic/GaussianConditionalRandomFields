function [ CRFData ] = createCRFstruct( N, T, steps_to_predict, maxiter, ...
    y, x, similarities, predictors, variances, confidenceGuesses,...
    alpha_features, beta_features )
%   This funtion creates a structure needed for GCRF training

CRFData = struct;
CRFData.N = N;
CRFData.maxiter = maxiter;

CRFData.Ttr = T(2)-T(1)+1;
CRFData.Ttest = steps_to_predict;
CRFData.T = CRFData.Ttr + CRFData.Ttest;

ytr = y(:, T(1):T(2));
ytest = y(:, T(2)+1 : T(2) + steps_to_predict);

CRFData.label = [~isnan(ytr) zeros(size(ytest))];
CRFData.label = logical(CRFData.label(:));

CRFData.predictors = predictors;
CRFData.noPredictors = length(predictors);

for i = 1:length(variances)
    variances{i}(variances{i}==0) = 0.0001;
end

CRFData.variances = variances;
CRFData.confidenceGuesses = confidenceGuesses;
CRFData.similarities = similarities;

CRFData.noAlphas = length(predictors);
CRFData.noAlphas_new = length(predictors);
CRFData.noAlphas_uf = length(predictors);
CRFData.noBetas = length(similarities);
CRFData.noBetas_new = length(similarities);
CRFData.noBetas_uf = length(similarities);

CRFData.alpha_features = alpha_features;
CRFData.beta_features = beta_features;
CRFData.theta_alpha = nan(1, CRFData.noAlphas*(length(CRFData.alpha_features)+1));
CRFData.theta_beta = nan(1, CRFData.noBetas*length(CRFData.beta_features));

CRFData.y_original = y;
CRFData.y = y(:);
CRFData.x = x;

if CRFData.noBetas == 1
    CRFData.betaMatrix{1} = blkdiag(CRFData.similarities{1}{:});
    
    CRFData.laplacianMatrix = blkdiag(CRFData.similarities{1}{:});
    CRFData.laplacianMatrix = diag(sum(CRFData.laplacianMatrix, 2)) - ...
        CRFData.laplacianMatrix;
else
    %When you wish to include several graph similarities,
    %similarities needs to be a structure that contains similarity matrices 
    %of dimensions (N,N,T).
    for i=1:CRFData.noBetas
        for t = 1:CRFData.Ttr + CRFData.Ttest
            blocks{t} =  CRFData.similarities{i}{t};
        end
        CRFData.betaMatrix{i} = blkdiag(blocks{:});
    end
    
    for i=1:CRFData.noBetas
        for t = 1:CRFData.Ttr + CRFData.Ttest
            blocks{t} =  CRFData.similarities{i}{t};
        end
        CRFData.laplacianMatrix{i} = blkdiag(blocks{:});
        CRFData.laplacianMatrix{i} = diag(sum(CRFData.laplacianMatrix{i}, 2)) - ...
            CRFData.laplacianMatrix{i};
    end
end
end

