function [totalLikelihood, totalGradient, mu, Q] = objectiveCRF(u, CRFData)

nt = CRFData.N * CRFData.Ttr;
n = CRFData.N;
t = CRFData.Ttr;

% split vector u to find ualpha and ubeta
alpha = nan(CRFData.noAlphas, n);
for i = 1:CRFData.noAlphas
   alpha(i,:) = exp(u((i-1)*n + 1 : (i-1)*n + n));
end

% alpha = reshape(exp(u(1 : CRFData.noAlphas_new*n)), CRFData.noAlphas_new, n);
beta = exp(u(CRFData.noAlphas*n +1 : end));

% calculate precision matrix
% [Q1, Q2, b] = calcPrecision(ualpha, ubeta, CRFData, 'training');

totalLikelihood = 0;
totalGradient = zeros(size(u));

for nts = 1:t
    
    Q1 = sparse(n, n);
    b = zeros(n, 1);
    
    for i=1:CRFData.noAlphas
        bb =  2 * alpha(i,:)' .* CRFData.predictors{i}(:, nts);
        b = b + bb(:);

        Q1 = Q1 + diag(alpha(i, :)');
        
    end;
    
    Q2 = sparse(n,n);
    
    for i=1:CRFData.noBetas
        Q2 = Q2 - beta(i) * CRFData.similarities{i}{nts};
    end;
    
    aux = full(sum(Q2,2));
    Q2diag = spconvert([(1:length(aux))', (1:length(aux))', aux]);
    Q2 = Q2 - Q2diag;
    
    Q = 2*(Q1 + Q2);
    
    % calculate precision matrix for labeled data in the training set
    Qll = Q(CRFData.label(n*(nts-1) + 1 : n*nts), CRFData.label(n*(nts-1) + 1 : n*nts));
    bl = b(CRFData.label(n*(nts-1) + 1 : n*nts));
    ylabel = CRFData.y(CRFData.label(n*(nts-1) + 1 : n*nts));
    
    % calculate likelihood for training data
    RR= chol(Qll);
    mu = RR\(RR'\bl);
    
    regular = CRFData.lambdaAlpha*(1/2)*(1/n)*sum(sum(alpha.^2)) + CRFData.lambdaBeta*(1/2)*sum(beta.^2);
    
    % calculate likelihood
    likelihood = - sum(log(diag(RR))) + 0.5*(ylabel-mu)' * Qll * (ylabel-mu) + regular;
    
    % calculate derivatives
    dualpha = derivative_alpha(alpha, Qll, mu, ylabel, CRFData, nts);
    dubeta = derivative_beta(beta, Qll, mu, ylabel, CRFData, nts) - ...
        CRFData.lambdaBeta*beta;
    
    for i = 1:CRFData.noAlphas_new
       dualphaFlatten((i-1)*n + 1 : (i-1)*n + n) = dualpha(i,:);
    end
    
    g = [-dualphaFlatten -dubeta];
    
    totalLikelihood = totalLikelihood + likelihood;
    totalGradient = totalGradient + g;
    
end

end
