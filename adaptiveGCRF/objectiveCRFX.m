function [totalLikelihood, totalGradient, mu, Q] = objectiveCRFX(u,Data)
n = Data.N;
t = Data.Ttr;

%% split vector u to find ualpha and ubeta
ualpha = u(1:Data.nalpha);
ubeta = u(Data.nalpha+1:Data.nalpha+Data.nbeta);
thetas_Rk = u(Data.nalpha+Data.nbeta+1:Data.nalpha+Data.nbeta+Data.nthetas_Rk);
thetas_Sl = u(Data.nalpha+Data.nbeta+Data.nthetas_Rk+1:end);

%% reshape alpha in #UnstrPred's x N
% % % % % % alpha = nan(Data.noUnstrPreds, Data.N);
% % % % % % alpha = reshape(exp(u), [], n);
% % % % % % % for i = 1:CRFData.noAlphas
% % % % % % %    alpha(i,:) = exp(u((i-1)*n + 1 : (i-1)*n + n));
% % % % % % % end
% % % % % %
% % % % % % alpha = nan(CRFData.noAlphas, Data.N);
% % % % % % alpha = reshape(exp(u), [],n);

% % % % %% calculate precision matrix
% % % % [Q1, Q2, b] = calcPrecisionX(ualpha, ubeta, thetas_Rk, thetas_Sl, Data, 'training');
% % % % Q = 2*(Q1 + Q2);
% % % %
% % % % %% calculate precision matrix for labeled data in the training set
% % % % Qll = Q(Data.label(1:nt), Data.label(1:nt));
% % % % bl = b(Data.label(1:nt));
% % % % ylabel = Data.y(Data.label(1:nt));
% % % %
% % % % %% calculate likelihood for training data
% % % % RR = chol(Qll);
% % % % mu = Qll\bl;
% % % %
% % % % regulralpha = 50*pi;
% % % % regularbeta = pi;
% % % % regular = regulralpha*(1/2)*exp(ualpha)^2+regularbeta*(1/2)*exp(ubeta)^2;
% % % %
% % % % f = calcLikelihoodX(RR, Qll, ylabel, mu) + regular;
% % % %
% % % % %% calculate first derivatives with respect to parameters
% % % % dualpha = derivativeAlphaX(Data,Q1,Q2,ylabel,mu,ualpha) - regulralpha*exp(ualpha);
% % % % dubeta = derivativeBetaX(Data,Q1,Q2,ylabel,mu,ubeta) - regularbeta*exp(ubeta);
% % % % thetas_Rk = u_derivativeP_Rk(ylabel, mu, ualpha) * u_derivativeRk_ThetaRk_LR(Data, thetas_Rk);
% % % % % % % thetas_Sl = s_derivativeP_Sl(dP_Sl, dSl_ThetaSl) * s_derivativeSl_ThetaSl_EDScale(Data , thetaSl);
% % % %
% % % % g =[-dualpha -dubeta -thetas_Rk -thetas_Sl];

totalLikelihood = 0;
totalGradient = zeros(size(u));

for nts = 1:t
    %     nts
    Q1 = sparse(n,n);
    b = zeros(n,1);
    
    % reordering input attributes for training
    %     xx = Data.x(:, Data.xunstr, nts);
    %     X = reshape(ipermute(xx,[1 3 2]), [], length(Data.xunstr));
    %     X = [ones(size(X,1),1), X];
    X=Data.X_unstr{nts};
    % ako je unstr linerana regresija
    R_nts=zeros(n, Data.nalpha);
    Sl_nts={};
    for i=1:Data.nalpha
        R_nts(:,i)=X *thetas_Rk';
        bb =  2 * exp(ualpha(i)) * R_nts(:,i);
        b = b + bb(:);
        Q1 = Q1 + exp(ualpha(i)) * speye(n,n);
    end
    
    Q2 = sparse(n,n);
    
    %     for i=1:Data.nbeta
    %         Q2 = Q2 - exp(ubeta(i)) * Data.similarities{i}{nts};
    Sl_nts{1}=calcSimilarityX_GaussianKernel(thetas_Sl, Data, nts);
    Q2 = Q2 - exp(ubeta(1)) * Sl_nts{1};
    %     end
    
    aux = sum(Q2,2);
    Q2diag = spconvert([(1:length(aux))', (1:length(aux))', aux]);
    Q2 = Q2 - Q2diag;
    
    Q = 2*(Q1 + Q2);
    
    % calculate precision matrix for labeled data in the training set
    Qll = Q(Data.label(n*(nts-1) + 1 : n*nts), Data.label(n*(nts-1) + 1 : n*nts));
    bl = b(Data.label(n*(nts-1) + 1 : n*nts));
    ylabel = Data.y_flat(Data.label(n*(nts-1) + 1 : n*nts));
    
    
    % calculate likelihood for training data
    %     RR= chol(Qll);
    %     mu = RR\(RR'\bl);
    Qll_inv = inv(Qll);
    mu = Qll\bl;
    
    % % % %     plot(mse(mu-ylabel))
    
    regular = Data.lambdaAlpha*(1/2)*exp(ualpha)^2 + Data.lambdaBeta*(1/2)*exp(ubeta)^2+ sum( Data.lambdaR*(0.5).*thetas_Rk.^2) +sum( Data.lambdaS*(0.5).*thetas_Sl.^2);
    % calculate likelihood
    likelihood = - sum(log(diag(Qll_inv))) + 0.5*(ylabel-mu)' * Qll * (ylabel-mu) + regular;
    
    % calculate derivatives
    dualpha = derivativeAlphaX(Data,Qll, R_nts,ylabel,mu,ualpha,nts) - Data.lambdaAlpha*exp(ualpha);
    dubeta = derivativeBetaX(Data,Qll,Sl_nts,ylabel,mu,ubeta,nts) - Data.lambdaBeta*exp(ubeta);
    dtheta_Rk = u_derivativeThetaRk( Data, ualpha, ylabel, mu, R_nts, nts )-Data.lambdaR*thetas_Rk;
    dtheta_Sl = s_derivativeThetaSl(Data, Qll,  Sl_nts , ylabel, mu, ubeta, thetas_Sl, nts)-Data.lambdaS*thetas_Sl;
    %     dtheta_Sl = thetas_Sl;
    
    g = [-dualpha -dubeta -dtheta_Rk -dtheta_Sl];
    
    totalLikelihood = totalLikelihood + likelihood;
    totalGradient = totalGradient + g;
    
end
