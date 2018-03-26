function dualpha = derivative_alpha(ualpha, Qll, mu, ylabel, CRFData)

nt = CRFData.N * CRFData.Ttr;
NT = CRFData.N * CRFData.T;
dualpha = nan(1, CRFData.noAlphas);

for i = 1 : CRFData.noAlphas
    
    dQ = 2 * speye(NT, NT);
    dQll = dQ(CRFData.label(1:nt), CRFData.label(1:nt));
    db = 2 * CRFData.predictors{i}(:, 1:CRFData.Ttr);
    db = db(:);
    dbl = db(CRFData.label(1:nt));
    %     dualpha(i)= (-1/2 * (ylabel - mu)' * dQll * (ylabel - mu) + (dbl' - mu' * dQll) * (ylabel - mu) + 1/2 * computeTraceParallel(Qll, dQll)) * exp(ualpha(i));
    dualpha(i)= (-1/2 * (ylabel - mu)' * dQll * (ylabel - mu) + (dbl' - mu' * dQll) * (ylabel - mu) + 1/2 * blokTraceInvMatrix(Qll, dQll, CRFData.N,CRFData.Ttr)) * exp(ualpha(i));
    
    
end;