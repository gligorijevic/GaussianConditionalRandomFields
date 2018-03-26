function dubeta = derivativeBetaX(Data,Q,Sl_nts,ylabel,mu,ubeta, nts)
n = Data.N;
dubeta = nan(1,Data.nbeta);

for i=1:Data.nbeta
%     dQ = -2 * Data.similarities{i}{nts};
    dQ = -2 * Sl_nts{i};
    aux = sum(dQ,2);
    
    dQdiag = spconvert([(1:length(aux))', (1:length(aux))', aux]);
    dQ = dQ - dQdiag;
    dQll = dQ(Data.label(n*(nts-1) + 1 : n*nts),Data.label(n*(nts-1) + 1 : n*nts));

    dubeta(i)= (-0.5*(ylabel + mu)' * dQll * (ylabel - mu) + ...
         0.5 * trace(dQll/Q))* exp(ubeta(i));
        %1/2 * betaTraceInvMatrix(Qll, dQll)) * exp(ubeta(i));
    
end;
end

function [prodtraceTemp] = betaTraceInvMatrix(matrix,dMatrix)

prodtraceTemp=0;

rr = chol(matrix);
irr=inv(rr);
invMatrixblock=irr*irr';
prodtraceTemp= prodtraceTemp+trace(invMatrixblock*dMatrix);

end