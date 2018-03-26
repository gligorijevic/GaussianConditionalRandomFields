function dualpha = derivative_alpha_new( alpha, Qll, mu, ylabel, CRFData, nts )

n = CRFData.N;
dualpha = nan(CRFData.noAlphas_new, n);

for i = 1 : CRFData.noAlphas_new
    
    dQ = 2 * speye(n, n);
    dQll = dQ(CRFData.label(n*(nts-1) + 1 : n*nts), CRFData.label(n*(nts-1) + 1 : n*nts));
    db = 2 * CRFData.predictors{i}(:, nts);
    db = db(:);
    dbl = db(CRFData.label(n*(nts-1) + 1 : n*nts));
    
    perdictor_sigma = CRFData.variances{i}(:, nts);
    perdictor_sigma = perdictor_sigma(:);
    
    dualpha(i, :) = (...
        (-0.5 * (ylabel - mu)' * dQll * (ylabel - mu) + (dbl' - mu' * dQll) * (ylabel - mu) + ...
        0.5 * alphaTraceInvMatrix(Qll, dQll)) * ...
        (alpha(i,:)' ./ CRFData.variances{i}(:, nts)))' - ...
        CRFData.lambdaAlpha*sum((alpha(i,:)' ./ CRFData.variances{i}(:, nts)).^2) * CRFData.confidenceGuesses{i};
    
end;
end

function [prodtraceTemp] = alphaTraceInvMatrix(matrix,dMatrix)
prodtraceTemp=0;

rr = chol(matrix);
irr=inv(rr);
invMatrixblock=irr*irr';
prodtraceTemp= prodtraceTemp+trace(invMatrixblock*dMatrix);

end
