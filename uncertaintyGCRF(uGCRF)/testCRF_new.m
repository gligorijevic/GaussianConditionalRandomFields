function [ predictionCRF, Sigma, Variance, predictionCRF_all, Sigma_all, Variance_all ] = testCRF_new( CRFData )
NT = CRFData.N * CRFData.T;
n = CRFData.N;

% alpha = reshape(exp(CRFData.ualpha_new), CRFData.noAlphas_new, n);
for i = 1:CRFData.noAlphas_new
   alpha(i,:) = exp(CRFData.ualpha_new((i-1)*n + 1 : (i-1)*n + n));
end
beta = exp(CRFData.ubeta_new);


% [Q1 Q2 b] = calcPrecision_new(CRFData.ualpha_new, CRFData.ubeta_new, CRFData, 'test');
% Q = 2*(Q1 + Q2);

allCovarianceMatrices = {};
fullMeanVector = [];

for nts = 1:CRFData.T
    
    Q1 = sparse(n, n);
    b = zeros(n, 1);
    
    for i = 1:CRFData.noAlphas_new
        
        bb =  2 * (alpha(i,:)' ./ CRFData.variances{i}(:, nts)) .* CRFData.predictors{i}(:, nts)...
        * CRFData.confidenceGuesses{i};
        b = b + bb(:);
        
        alpha_norm = (alpha(i, :)' ./ CRFData.variances{i}(:, nts))...
            * CRFData.confidenceGuesses{i};
        alpha_norm = alpha_norm(:);
        Q1 = Q1 + diag(alpha_norm);
        
    end;
    
    Q2 = sparse(n,n);
    
    for i=1:CRFData.noBetas_new
        
        Q2 = Q2 - beta(i) * CRFData.similarities{i}{nts};
        
    end;
    aux = full(sum(Q2,2));
    Q2diag = spconvert([(1:length(aux))', (1:length(aux))', aux]);
    Q2 = Q2 - Q2diag;
    
    Q = 2*(Q1 + Q2);
    
    %     Qll = Q(CRFData.label(n*(nts-1) + 1 : n*nts), CRFData.label(n*(nts-1) + 1 : n*nts));
    %     bl = b(CRFData.label(n*(nts-1) + 1 : n*nts));
    %     ylabel = CRFData.y(CRFData.label(n*(nts-1) + 1 : n*nts));
    %
    %     RR= chol(Qll);
    %     mu = Qll\bl;
    
    RR= chol(Q);
    %     mu = Qll\bl;
    munew = RR\(RR'\b);
    
    %     munew = Q\b;
    Sigma = RR\(RR'\speye(n));
    
    %     Variance = diag(Sigma);
    %
    %     predictionCRF_all = munew;
    %     Variance_all = Variance;
    
    % take parts of precision matrix for conditional prediction
    %     Ql = Q(~CRFData.label(n*(nts-1) + 1 : n*nts), ~CRFData.label(n*(nts-1) + 1 : n*nts));
    %     Qul = Q(~CRFData.label(n*(nts-1) + 1 : n*nts), CRFData.label(n*(nts-1) + 1 : n*nts));
    
    %     predictionCRF = munew(~CRFData.label(n*(nts-1) + 1 : n*nts)) + ...
    %         Ql\(Qul*(CRFData.y(CRFData.label(n*(nts-1) + 1 : n*nts)) - ...
    %         munew(CRFData.label(n*(nts-1) + 1 : n*nts))));
    % predictionCRF(predictionCRF<0)=0;
    
    allCovarianceMatrices{nts} = Sigma;
    fullMeanVector = [fullMeanVector; munew];
    
end

predictionCRF_all = fullMeanVector;
predictionCRF = fullMeanVector(end - CRFData.N * CRFData.Ttest + 1 : end);
Sigma_all = blkdiag(allCovarianceMatrices{:});
Variance_all = diag(Sigma_all);
Variance = diag(Sigma_all(end - CRFData.N * CRFData.Ttest + 1 : end, end - CRFData.N * CRFData.Ttest + 1 : end));
Sigma = Sigma_all(end - CRFData.N *CRFData.Ttest + 1 : end, end - CRFData.N *CRFData.Ttest + 1 : end);

end