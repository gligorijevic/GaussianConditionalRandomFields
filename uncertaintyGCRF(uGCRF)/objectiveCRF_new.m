function [ totalLikelihood, totalGradient, mu, Q ] = objectiveCRF_new( u, CRFData )
n = CRFData.N; 
t = CRFData.Ttr;
nt = CRFData.N * CRFData.Ttr;

% split vector u to find ualpha and ubeta
alpha = nan(CRFData.noAlphas_new, CRFData.N);
for i = 1:CRFData.noAlphas_new
   alpha(i,:) = exp(u((i-1)*n + 1 : (i-1)*n + n));
end

% alpha = reshape(exp(u(1 : CRFData.noAlphas_new*n)), CRFData.noAlphas_new, n);
beta = exp(u(CRFData.noAlphas_new*n +1 : end));

% calculate precision matrix
% [Q1, Q2, b] = calcPrecision_new(ualpha, ubeta, CRFData, 'training');

totalLikelihood = 0;
totalGradient = zeros(size(u));

for nts = 1 : t
    
    Q1 = sparse(n, n);
    b = zeros(n, 1);
    
    for i = 1:CRFData.noAlphas_new
        
        bb =  2 * (alpha(i,:)' ./ CRFData.variances{i}(:, nts)) ...
             .* CRFData.predictors{i}(:, nts) * CRFData.confidenceGuesses{i};
        b = b + bb(:);
        
        alpha_norm = (alpha(i, :)' ./ CRFData.variances{i}(:, nts)) ...
            .* CRFData.confidenceGuesses{i};
        alpha_norm = alpha_norm(:);
        Q1 = Q1 + diag(alpha_norm);
        
    end;
    
    Q2 = sparse(n,n);
    
    for i=1:CRFData.noBetas_new
        Q2 = Q2 - beta(i) * CRFData.similarities{i}{nts};
    end;
    
    aux = full(sum(Q2,2));
    Q2diag = spconvert([(1:length(aux))', (1:length(aux))', aux]);
    Q2 = Q2 - Q2diag;
    
    Q = 2*(Q1 + Q2);
    
    Qll = Q(CRFData.label(n*(nts-1) + 1 : n*nts), CRFData.label(n*(nts-1) + 1 : n*nts));
    bl = b(CRFData.label(n*(nts-1) + 1 : n*nts));
    ylabel = CRFData.y(CRFData.label(n*(nts-1) + 1 : n*nts));
    
    RR= chol(Qll);
    mu = RR\(RR'\bl);
%     Qll_inv = inv(Qll);
%     mu = Qll\bl;
    
    regular = CRFData.lambdaAlpha*(1/2)*(1/n)*sum(sum(alpha.^2))+CRFData.lambdaBeta*(1/2)*sum(beta.^2);
    
    % calculate likelihood
    likelihood = - sum(log(diag(RR))) + 0.5*(ylabel-mu)' * Qll * (ylabel-mu) + regular;
    
    % calculate derivatives
    dualpha = derivative_alpha_new(alpha, Qll, mu, ylabel, CRFData, nts) ; %-regulralpha*exp(ualpha); %make it a matrix, k*p (p time steps for prediction) , also derive alpha with sigma of uns pred
    dubeta = derivative_beta_new(beta, Qll, mu, ylabel, CRFData, nts) - ...
        CRFData.lambdaBeta*beta; %-regularbeta*exp(ubeta);
    
    for i = 1:CRFData.noAlphas_new
       dualphaFlatten((i-1)*n + 1 : (i-1)*n + n) = dualpha(i,:);
    end
    
    g = [-dualphaFlatten -dubeta];
    
    totalLikelihood = totalLikelihood + likelihood;
    totalGradient = totalGradient + g;

end

end