function dualpha = derivativeAlphaX(Data,Q, R_nts, ylabel,mu,ualpha, nts)
n = Data.N;
dualpha = nan(1,Data.nalpha);


for i=1:Data.nalpha
    dQ = 2*speye(n, n);
    dQll = dQ(Data.label(n*(nts-1) + 1 : n*nts),Data.label(n*(nts-1) + 1 : n*nts));
    
    db = 2 * R_nts(:,i);
    %db = db(:);
    dbl = db(Data.label(n*(nts-1) + 1 : n*nts));
    dualpha(i, :)= (-0.5 * (ylabel - mu)' * dQll * (ylabel - mu) +  ...
        (dbl' - mu' * dQll) * (ylabel - mu) + ...
        0.5 * trace(dQll/Q)) * exp(ualpha(i));
        %0.5 * alphaTraceInvMatrix(Q, dQll)) * exp(ualpha(i));
end;
end

function [prodtraceTemp] = alphaTraceInvMatrix(matrix,dMatrix)
prodtraceTemp=0;

rr = chol(matrix);
irr=inv(rr);
invMatrixblock=irr*irr';
prodtraceTemp= prodtraceTemp+trace(invMatrixblock*dMatrix);

end