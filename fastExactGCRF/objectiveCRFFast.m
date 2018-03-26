function [likelihood, g, mu, Q] = objectiveCRFFast(u, CRFData)

nt = CRFData.N * CRFData.Ttr;

% split vector u to find ualpha and ubeta
ualpha = u(1 : CRFData.noAlphas);
ubeta = u(CRFData.noAlphas+1 : end);

% calculate precision matrix
[b, Q, Qinv] = calcPrecisionFast(ualpha, ubeta, CRFData, 'training');
% Q = 2*(Q1 + Q2);

% calculate precision matrix for labeled data in the training set
Qll = Q(CRFData.label(1:nt), CRFData.label(1:nt));
Qinvll = Qinv(CRFData.label(1:nt), CRFData.label(1:nt));
bl = b(CRFData.label(1:nt));
ylabel = CRFData.y(CRFData.label(1:nt));

% calculate likelihood for training data
% RR= chol(Qll);
% mu = Qll\bl;

mu = Qinvll*bl;

regular = sum(CRFData.lambdaAlpha*(1/2)*exp(ualpha).^2)+sum(CRFData.lambdaBeta*(1/2)*exp(ubeta).^2);

% calculate likelihood
likelihood = - sum(log(diag(Qll))) + 0.5*(ylabel-mu)' * Qll * (ylabel-mu) + regular;

% calculate derivatives
dualpha = derivative_alphaFast(ualpha, Qinvll, mu, ylabel, CRFData) - CRFData.lambdaAlpha*exp(ualpha);
dubeta = derivative_betaFast(ubeta, Qinvll, mu, ylabel, CRFData) - CRFData.lambdaBeta*exp(ubeta);

g = [-dualpha -dubeta];

end
