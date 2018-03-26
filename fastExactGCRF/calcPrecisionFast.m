function [ b, Q, Qinv ] = calcPrecisionFast( ualpha, ubeta, CRFData, str )
%CALCPRECISIONFAST

if (strcmp(str,'training')==1)
    t = CRFData.Ttr;
    D = CRFData.D;
    V = CRFData.V;
end;

if (strcmp(str,'test')==1)
    t = CRFData.Ttr + CRFData.Ttest;
    D = CRFData.DTest;
    V = CRFData.VTest;
end;

nt = CRFData.N * t;
b = zeros(nt, 1);

for i=1:CRFData.noAlphas
    bb =  2 * exp(ualpha(i)) * CRFData.predictors{i}(:, 1:t);
    b = b + bb(:);
end;

% [U,D,UT] = svd(CRFData.laplacianMatrix(1:nt, 1:nt));
% [V,D] = eig(CRFData.laplacianMatrix(1:nt, 1:nt));
Lambda = sum(exp(ualpha)).*eye(size(D)) + sum(exp(ubeta)).*D;
LambdaInv = 1./diag(Lambda);
LambdaInv = diag(LambdaInv);
Q = 2.*V*Lambda*V';
Qinv = .5.*V*LambdaInv*V';

end