function dubeta = derivative_beta(beta, Qll, mu, ylabel, CRFData, nts)

n = CRFData.N;
dubeta = nan(1, CRFData.noBetas);

for i = 1:CRFData.noBetas
    
    dQ = -2 * CRFData.similarities{i}{nts};
    aux = full(sum(dQ,2));
    dQdiag = spconvert([(1:length(aux))', (1:length(aux))', aux]);
    dQ = dQ - dQdiag;
    dQll = dQ(CRFData.label(n*(nts-1) + 1 : n*nts), CRFData.label(n*(nts-1) + 1 : n*nts));
    dubeta(i)= (-1/2*(ylabel + mu)' * dQll * (ylabel - mu) + ...
        1/2 * betaTraceInvMatrix(Qll, dQll)) * beta(i);
    
end;
end

function [prodtraceTemp] = betaTraceInvMatrix(matrix,dMatrix)

prodtraceTemp=0;

rr = chol(matrix);
irr=inv(rr);
invMatrixblock=irr*irr';
prodtraceTemp= prodtraceTemp+trace(invMatrixblock*dMatrix);

end