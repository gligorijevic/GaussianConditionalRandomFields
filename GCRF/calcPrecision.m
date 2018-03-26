function [Q1 Q2 b] = calcPrecision(ualpha, ubeta, CRFData, str)

if (strcmp(str,'training')==1)
    t = CRFData.Ttr;
end;

if (strcmp(str,'test')==1)
    t = CRFData.Ttr + CRFData.Ttest;
end;

nt = CRFData.N * t;
Q1 = sparse(nt, nt);
b = zeros(nt, 1);

for i=1:CRFData.noAlphas
    bb =  2 * exp(ualpha(i)) * CRFData.predictors{i}(:, 1:t);
    b = b + bb(:);
    Q1 = Q1 + exp(ualpha(i)) * speye(nt, nt); %mozda ovo speye usporava
end;

Q2 = sparse(nt,nt);

for i=1:CRFData.noBetas
    Q2 = Q2 - exp(ubeta(i)) * CRFData.betaMatrix{i}(1:nt, 1:nt);
end;

aux = full(sum(Q2,2));
Q2diag = spconvert([(1:length(aux))', (1:length(aux))', aux]);
Q2 = Q2 - Q2diag;