function [ CRFData ] = createCRFstruct( N, T, steps_to_predict, maxiter, ...
    y, x, similarities, predictors)
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

CRFData.similarities = similarities;

CRFData.noAlphas = length(predictors);
CRFData.noBetas = length(similarities);

CRFData.y_original = y;
CRFData.y = y(:);
CRFData.x = x;

if CRFData.noBetas == 1
    % for training
    CRFData.betaMatrix{1} = kron(eye(CRFData.Ttr, CRFData.Ttr), CRFData.similarities{1});
    
    
    CRFData.laplacianMatrix = CRFData.betaMatrix{1};
    CRFData.laplacianMatrix = diag(sum(CRFData.laplacianMatrix, 2)) - ...
        CRFData.laplacianMatrix;
    
    [CRFData.V, CRFData.D] = eig(CRFData.laplacianMatrix);
    
    %for testing
    CRFData.betaMatrixTest{1} = kron(eye(CRFData.Ttr+CRFData.Ttest, CRFData.Ttr+CRFData.Ttest), CRFData.similarities{1});
    
    
    CRFData.laplacianMatrixTest = CRFData.betaMatrixTest{1};
    CRFData.laplacianMatrixTest = diag(sum(CRFData.laplacianMatrixTest, 2)) - ...
        CRFData.laplacianMatrixTest;
    
    [CRFData.VTest, CRFData.DTest] = eig(CRFData.laplacianMatrixTest);
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

