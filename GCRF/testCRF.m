function [ predictionCRF, Sigma, Variance, predictionCRF_all, Sigma_all, Variance_all ]= testCRF(CRFData)

NT = CRFData.N * CRFData.T;

[Q1 Q2 b] = calcPrecision(CRFData.ualpha, CRFData.ubeta, CRFData, 'test');
Q = 2*(Q1 + Q2);

munew = Q\b;
Sigma = inv(Q);
Variance = diag(Sigma);

predictionCRF_all = munew;
Variance_all = Variance;

% take parts of precision matrix for conditional prediction
Q1 = Q(~CRFData.label(1:NT), ~CRFData.label(1:NT));
Qul = Q(~CRFData.label(1:NT), CRFData.label(1:NT));

predictionCRF = munew(~CRFData.label(1:NT)) + Q1\(Qul*(CRFData.y(CRFData.label(1:NT)) - munew(CRFData.label(1:NT))));



% predictionCRF(predictionCRF<0)=0;
predictionCRF = predictionCRF(end - CRFData.N * CRFData.Ttest + 1 : end);
Variance = Variance(end - CRFData.N * CRFData.Ttest + 1 : end);
Sigma_all = Sigma;
Sigma = Sigma(end - CRFData.N *CRFData.Ttest + 1 : end, end - CRFData.N *CRFData.Ttest + 1 : end);

end