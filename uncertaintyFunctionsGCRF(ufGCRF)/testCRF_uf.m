function [ predictionCRF, Sigma, Variance, predictionCRF_all, Sigma_all, Variance_all ] = testCRF_uf( CRFData )

NT = CRFData.N * CRFData.T;
nt = CRFData.N * CRFData.Ttr;
n = CRFData.N;
t = CRFData.Ttr;
m = size(CRFData.x, 2);

% split vector u to find ualpha and ubeta
theta_alpha = {};
for i = 1:CRFData.noAlphas_new
    theta_alpha{i} = nan(length(CRFData.alpha_features)+1, CRFData.N);
    for j = 1:length(CRFData.alpha_features)+1
        % elementi idu sledecim redosledom: prvo ide w0 za svih n cvorova
        % za prvi prediktor, pa w1 za svih n cvorova za prvi prediktor i
        % tako dalje a zatim w0 za sve cvorove za drugi prediktor i tako
        % dalje...
        theta_alpha{i}(j,:) = CRFData.ualpha_uf((i-1)*(j-1) + 1 : (i-1)*(j-1)*n + n);
    end
end

beta = exp(CRFData.ubeta_uf);

alpha = {};


% calculate precision matrix
% [Q1, Q2, b] = calcPrecision_uf(theta_alpha, theta_beta, CRFData, 'training');

allCovarianceMatrices = {};
fullMeanVector = [];

for nts = 1 : CRFData.T
    
    Q1 = sparse(n, n);
    b = zeros(n, 1);
    
    for i = 1:CRFData.noAlphas_uf
        alpha{i} = zeros(CRFData.N,1);
        alpha{i} = exp(sum([ones(CRFData.N,1), CRFData.x(:,CRFData.alpha_features,nts)] ...
            .* theta_alpha{i}(:,:)', 2));
        
        bb =  2 * alpha{i} .* CRFData.predictors{i}(:, nts);
        b = b + bb(:);
        
        Q1 = Q1 + diag(alpha{i});
    end;
    
    Q2 = sparse(n,n);
    
    for i=1:CRFData.noBetas_uf
        Q2 = Q2 - beta(i) * CRFData.similarities{i}{nts};
    end;
    
    aux = full(sum(Q2,2));
    Q2diag = spconvert([(1:length(aux))', (1:length(aux))', aux]);
    Q2 = Q2 - Q2diag;
    
    Q = 2*(Q1 + Q2);
    
% %     Qll = Q(CRFData.label(n*(nts-1) + 1 : n*nts), CRFData.label(n*(nts-1) + 1 : n*nts));
% %     bl = b(CRFData.label(n*(nts-1) + 1 : n*nts));
% %     ylabel = CRFData.y(CRFData.label(n*(nts-1) + 1 : n*nts));
    
    RR = chol(Q);
    munew = RR\(RR'\b);
    Sigma = RR\(RR'\speye(n));
    
    allCovarianceMatrices{nts} = Sigma;
    fullMeanVector = [fullMeanVector; munew];
    
end

predictionCRF_all = fullMeanVector;
predictionCRF = fullMeanVector(end - CRFData.N * CRFData.Ttest + 1 : end);
Sigma_all = blkdiag(allCovarianceMatrices{:});
Variance_all = diag(Sigma_all);
Variance = diag(Sigma_all(end - CRFData.N * CRFData.Ttest + 1 : end, end - CRFData.N * CRFData.Ttest + 1 : end));
Sigma = Sigma_all(end - CRFData.N *CRFData.Ttest + 1 : end, end - CRFData.N *CRFData.Ttest + 1 : end);

% % [Q1 Q2 b] = calcPrecision_uf(CRFData.ualpha_new, CRFData.ubeta_new, CRFData, 'test');
% % Q = 2*(Q1 + Q2);
% %
% % munew = Q\b;
% % Sigma = inv(Q);
% % Variance = diag(Sigma);
% %
% % predictionCRF_all = munew;
% % Variance_all = Variance;
% %
% % % take parts of precision matrix for conditional prediction
% % Q1 = Q(~CRFData.label(1:NT), ~CRFData.label(1:NT));
% % Qul = Q(~CRFData.label(1:NT), CRFData.label(1:NT));
% %
% % predictionCRF = munew(~CRFData.label(1:NT)) + Q1\(Qul*(CRFData.y(CRFData.label(1:NT)) - munew(CRFData.label(1:NT))));
% % % predictionCRF(predictionCRF<0)=0;
% %
% % predictionCRF = predictionCRF(end - CRFData.N * CRFData.Ttest + 1 : end);
% % Variance = Variance(end - CRFData.N * CRFData.Ttest + 1 : end);
% % Sigma_all = Sigma;
% % Sigma = Sigma(end - CRFData.N *CRFData.Ttest + 1 : end, end - CRFData.N *CRFData.Ttest + 1 : end);


end

