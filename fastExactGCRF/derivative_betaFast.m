function dubeta = derivative_betaFast(ubeta, Qinv, mu, ylabel, CRFData)

nt = CRFData.N * CRFData.Ttr;
dubeta = nan(1, CRFData.noBetas);

for i = 1:CRFData.noBetas
    
    %     M0 = sparse(diag(ones(1, CRFData.Ttr + CRFData.Ttest)));
    %     betaMatrix = kron(M0, CRFData.similarities{1}); %seljacki ubacena jedinicia, sredi
    
    dQ = -2 * CRFData.betaMatrix{i};
    aux = full(sum(dQ,2));
    dQdiag = spconvert([(1:length(aux))', (1:length(aux))', aux]);
    dQ = dQ - dQdiag;
    dQll = dQ(CRFData.label(1:nt), CRFData.label(1:nt));
%     dubeta(i)= (-1/2*(ylabel + mu)' * dQll * (ylabel - mu) + 1/2 * computeTraceParallel(Qll,dQll)) * exp(ubeta(i));
%     dubeta(i)= (-1/2*(ylabel + mu)' * dQll * (ylabel - mu) + 1/2 * blokTraceInvMatrix(Qll, dQll, CRFData.N,CRFData.Ttr)) * exp(ubeta(i));
    dubeta(i)= (-1/2*(ylabel + mu)' * dQll * (ylabel - mu) + 1/2 * calcTraceInvFast(Qinv, dQll, CRFData.N,CRFData.Ttr)) * exp(ubeta(i));
        
end;