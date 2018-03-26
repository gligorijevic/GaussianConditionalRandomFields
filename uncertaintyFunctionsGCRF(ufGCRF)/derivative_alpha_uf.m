function dualpha = derivative_alpha_uf( alpha, iterAlpha, Qll, mu, ylabel, CRFData, nts )

n = CRFData.N;
dualpha = nan(1, n);

% for i = 1 : CRFDaRta.noAlphas_new
    
    dQ = 2 * speye(n, n);
    dQll = dQ(CRFData.label(n*(nts-1) + 1 : n*nts), CRFData.label(n*(nts-1) + 1 : n*nts));
    db = 2 * CRFData.predictors{iterAlpha}(:, nts);
    db = db(:);
    dbl = db(CRFData.label(n*(nts-1) + 1 : n*nts));
        
    dualpha(1, :) = ...
        (-0.5 * (ylabel - mu)' * dQll * (ylabel - mu) + (dbl' - mu' * dQll) * (ylabel - mu) + ...
        0.5 * alphaTraceInvMatrix(Qll, dQll)) * ...
        alpha';
    
% end;
end

function [prodtraceTemp] = alphaTraceInvMatrix(matrix,dMatrix)
prodtraceTemp=0;

rr = chol(matrix);
irr=inv(rr);
invMatrixblock=irr*irr';
prodtraceTemp= prodtraceTemp+trace(invMatrixblock*dMatrix);

end