function [ likelihood, g, mu, Q ] = objectiveCRF_uf( u, CRFData )

nt = CRFData.N * CRFData.Ttr;
n = CRFData.N;
t = CRFData.Ttr;
m = size(CRFData.x, 2);

% split vector u to find ualpha and ubeta
% theta_alpha = u(1 : length(CRFData.theta_alpha));
% theta_beta = u(length(CRFData.theta_alpha)+1 : end);

% split vector u to find ualpha and ubeta
theta_alpha = {};
% length(CRFData.theta_alpha) * CRFData.noAlphas_uf * CRFData.N
for i = 1:CRFData.noAlphas_new
    theta_alpha{i} = nan(length(CRFData.alpha_features)+1, CRFData.N);
    for j = 1:length(CRFData.alpha_features)+1
        % elementi idu sledecim redosledom: prvo ide w0 za svih n cvorova
        % za prvi prediktor, pa w1 za svih n cvorova za prvi prediktor i
        % tako dalje a zatim w0 za sve cvorove za drugi prediktor i tako
        % dalje...
        theta_alpha{i}(j,:) = u((i-1)*(j-1) + 1 : (i-1)*(j-1)*n + n);
    end
end

beta = exp(u((length(CRFData.alpha_features)+1) * CRFData.noAlphas_uf * CRFData.N+1:end));
% % reshape(exp(u(CRFData.noAlphas_new*n +1 : end)), length(CRFData.theta_beta), CRFData.noBetas_uf);
% for i = 1:CRFData.noBetas
%     theta_beta{i} = u(length(CRFData.theta_alpha) * CRFData.noAlphas_uf * CRFData.N + ...
%         (i-1)*length(CRFData.theta_beta) + 1 :...
%         length(CRFData.theta_alpha) * CRFData.noAlphas_uf * CRFData.N + ...
%         (i-1)*length(CRFData.theta_beta) + length(CRFData.theta_beta));
% end

alpha = {};


% calculate precision matrix
% [Q1, Q2, b] = calcPrecision_uf(theta_alpha, theta_beta, CRFData, 'training');

totalLikelihood = 0;
totalGradient = zeros(size(u));

for nts = 1 : t
    
    Q1 = sparse(n, n);
    b = zeros(n, 1);
    
    for i = 1:CRFData.noAlphas_uf
        alpha{i} = zeros(CRFData.N,1);
        alpha{i} = exp(sum([ones(CRFData.N,1), CRFData.x(:,CRFData.alpha_features,nts)] ...
            .* theta_alpha{i}(:,:)', 2));
        
        bb =  2 * alpha{i} .* CRFData.predictors{i}(:, nts);
        b = b + bb(:);
        
        Q1 = Q1 + diag(alpha{i});
    end;
    
    Q2 = sparse(n,n);
    
    for i=1:CRFData.noBetas_uf
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
    
    sumAlpha = 0;
    sumBeta = 0;
    for i = 1:length(alpha)
        sumAlpha = sum(sum(alpha{i}.^2));
    end
    
    %     for i = 1:length(beta)
    %         sumBeta = sum(sum(beta{i}.^2));
    %     end
    
    regular = CRFData.lambdaAlpha*(1/2)*(1/n)*sumAlpha + CRFData.lambdaBeta*(1/2)*sum(beta);
    
    % calculate likelihood
    likelihood = - sum(log(diag(RR))) + 0.5*(ylabel-mu)' * Qll * (ylabel-mu) + regular;
    
    % calculate derivatives
    d_theta_ualpha = {};
    for i = 1:CRFData.noAlphas_uf
        d_theta_ualpha{i} = repmat(derivative_alpha_uf(alpha{i}, i, Qll, mu, ylabel, CRFData, nts)',1,length(CRFData.alpha_features)+1)...
            .*derivative_theta_alpha(mu, ylabel, CRFData, nts);
        regularization_alpha = CRFData.lambdaAlpha*(1/2)*(1/n)*sum(sum((theta_alpha{i})));
        d_theta_ualpha{i} = d_theta_ualpha{i} - regularization_alpha;
    end
    %     theta_beta = derivative_beta_uf(theta_beta, Qll, mu, ylabel, CRFData, nts)...
    %         .*derivative_theta_beta(mu, ylabel, CRFData, nts)';
    %     regularization_beta = CRFData.regularbeta*(1/2)*(1/n)*exp(theta_beta);
    %     regularization_beta = regularization_beta(:)';
    %     theta_beta = theta_beta(:)' - regularization_beta;
    dubeta = derivative_beta(beta, Qll, mu, ylabel, CRFData, nts) - ...
        CRFData.lambdaBeta*beta;
    
    
    dualphaFlatten = [];
    for i = 1:CRFData.noAlphas_uf
        for j = 1:length(CRFData.alpha_features)+1
            dualphaFlatten = [dualphaFlatten, d_theta_ualpha{i}(:,j)'];
        end
    end
    
    g = [-dualphaFlatten -dubeta];
    
    totalLikelihood = totalLikelihood + likelihood;
    totalGradient = totalGradient + g;
    
end

end