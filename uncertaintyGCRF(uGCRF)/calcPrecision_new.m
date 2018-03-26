function [ Q1, Q2, b ] = calcPrecision_new( ualpha, ubeta, CRFData, str )

if (strcmp(str,'training')==1)
    t = CRFData.Ttr;
end;

if (strcmp(str,'test')==1)
    t = CRFData.Ttr + CRFData.Ttest;
end;

nt = CRFData.N * t;
Q1 = sparse(nt, nt);
b = zeros(nt, 1);

for i = 1:CRFData.noAlphas_new
    
    bb =  2 * (exp(ualpha(i)) ./ CRFData.variances{i}(:, 1:t)) ...
        * CRFData.confidenceGuesses{i} .* CRFData.predictors{i}(:, 1:t);
    b = b + bb(:);
    
    alpha_norm = (exp(ualpha(i)) ./ CRFData.variances{i}(:, 1:t))...
        * CRFData.confidenceGuesses{i};
    alpha_norm = alpha_norm(:);
    Q1 = Q1 + diag(alpha_norm);
    
end;

Q2 = sparse(nt,nt);

for i=1:CRFData.noBetas_new
    
    Q2 = Q2 - exp(ubeta(i)) * CRFData.betaMatrix{i}(1:nt, 1:nt);
    
    aux = full(sum(Q2,2));
    Q2diag = spconvert([(1:length(aux))', (1:length(aux))', aux]);
    Q2 = Q2 - Q2diag;
    
end;

end

