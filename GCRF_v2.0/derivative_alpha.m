function dualpha = derivative_alpha(alpha, Qll, mu, ylabel, CRFData, nts)
n = CRFData.N;
dualpha = nan(CRFData.noAlphas, n);

for i = 1 : CRFData.noAlphas
    
    dQ = 2 * speye(n, n);
    dQll = dQ(CRFData.label(n*(nts-1) + 1 : n*nts), CRFData.label(n*(nts-1) + 1 : n*nts));
    db = 2 * CRFData.predictors{i}(:, nts);
    db = db(:);
    dbl = db(CRFData.label(n*(nts-1) + 1 : n*nts));
    dualpha(i, :)= (-1/2 * (ylabel - mu)' * dQll * (ylabel - mu) +...
        (dbl' - mu' * dQll) * (ylabel - mu) + ...
        1/2 * alphaTraceInvMatrix(Qll, dQll))...
        * alpha(i) - ...
        CRFData.lambdaAlpha*sum((alpha(i,:)).^2);
    
end;
end

function [prodtraceTemp] = alphaTraceInvMatrix(matrix,dMatrix)
prodtraceTemp=0;

rr = chol(matrix);
irr=inv(rr);
invMatrixblock=irr*irr';
prodtraceTemp= prodtraceTemp+trace(invMatrixblock*dMatrix);

end