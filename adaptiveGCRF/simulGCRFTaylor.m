function [ mu_final, sigma_final, SigmaX ] = simulGCRFTaylor( Data )
%% Simulation of the GCRF model, where the output variance is propagated
% using Taylor approximation.

n = Data.N;
d = length(Data.xunstr);

ualpha = Data.u(1:Data.nalpha);
ubeta = Data.u(Data.nalpha+1:Data.nalpha+Data.nbeta);
thetas_Rk = Data.u(Data.nalpha+Data.nbeta+1:Data.nalpha+Data.nbeta+Data.nthetas_Rk);
thetas_Sl = Data.u(Data.nalpha+Data.nbeta+Data.nthetas_Rk+1:end);

mu_final = zeros(n, Data.Ttr+Data.Ttest);
sigma_final = zeros(n,n,Data.Ttr+Data.Ttest);

% First obtain predictions on training time steps
for nts = 1:Data.Ttr
    Q1 = sparse(n, n);
    b = zeros(n, 1);
    
    X=Data.X_unstr{nts};
    R_nts=zeros(n, Data.nalpha);
    Sl_nts={};
    
    for i=1:Data.nalpha
        R_nts(:,i)=X *thetas_Rk';
        bb =  2 * exp(ualpha(i)) * R_nts(:,i);
        b = b + bb(:);
        
        Q1 = Q1 + exp(ualpha(i)) * speye(n,n);
    end;
    
    Q2 = sparse(n,n);
    Sl_nts{1}=calcSimilarityX_GaussianKernel(thetas_Sl, Data, nts);
    Q2 = Q2 - exp(ubeta(1)) * Sl_nts{1};
    
    aux = sum(Q2,2);
    Q2diag = spconvert([(1:length(aux))', (1:length(aux))', aux]);
    Q2 = Q2 - Q2diag;
    
    Q = 2*(Q1 + Q2);
    
    RR= chol(Q);
    munew = RR\(RR'\b);
    
    Sigma = RR\(RR'\speye(n));
    
    mu_final(:,nts) = munew;
    sigma_final(:,:,nts) = Sigma;
    
end

for i = 1:n
    SigmaX{i} = zeros(d,d);
end

% Predict one step ahead
correctionTerm = zeros(n,n); % just for testing purposes
[muGCRF, sigmaGCRF, derivMu, derivSigma, correctionTerm] = gcrfTaylorPredict(Data, SigmaX,correctionTerm, Data.Ttr + 1);
mu_final(:,Data.Ttr+1) = muGCRF;
sigma_final(:,:,Data.Ttr+1) = sigmaGCRF;

% Now update information in the input on the obtained predictions and
% iteratively repeat predicting process with including predictions as
% inputs for future predictions
for step_ahead = 2:Data.Ttest
    
    nts = Data.Ttr + step_ahead;
    
    for node_idx = 1:n
        % xtest = [xtest(2:lag) mu(i-1) test(i,lag+1:end)];
        
        % Update validation data (delayed outputs y first, previous 
        % time-step prediction, and input variables last)
        % 
        % We assume that ustructured predictor and similarity meaure might
        % account for different inputs, however in our experiments they use
        % same inputs (with addition that linear regression has a 1 for the
        % intercept term).
        Data.X_all{nts}(node_idx,:) = ...
            [Data.X_all{nts}(node_idx,2:Data.lag) muGCRF(node_idx) ...
            Data.X_all{nts}(node_idx,Data.lag+1:end)];
        Data.X_unstr{nts}(node_idx,:) = ...
            [1 Data.X_unstr{nts}(node_idx,3:Data.lag) muGCRF(node_idx) ...
            Data.X_unstr{nts}(node_idx,Data.lag+1:end)];
        Data.X_sim{nts}(node_idx,:) = ...
            [Data.X_sim{nts}(node_idx,2:Data.lag) muGCRF(node_idx) ...
            Data.X_sim{nts}(node_idx,Data.lag+1:end)];
        
        % Updating covariance matrix of inputs. Diagonal elements are the 
        % estimated variance when predicting. Off-diagonal elements are
        % approximated as well and they are calculated for each new
        % prediciton step, old covariance vectors are moved one step back.
        covXY = derivMu(node_idx,:)*SigmaX{node_idx};
        SigmaX{node_idx}(1:Data.lag-1,1:Data.lag-1) = ...
            SigmaX{node_idx}(2:Data.lag,2:Data.lag);
        SigmaX{node_idx}(Data.lag,Data.lag) = sigmaGCRF(node_idx, node_idx);
        SigmaX{node_idx}(1:Data.lag-1,Data.lag) = covXY(2:Data.lag)';
        SigmaX{node_idx}(Data.lag,1:Data.lag-1) = covXY(2:Data.lag);
        
    end
    
    [muGCRF, sigmaGCRF, derivMu, derivSigma, correctionTerm] = gcrfTaylorPredict(Data, SigmaX, correctionTerm, nts);
    
    mu_final(:,Data.Ttr+step_ahead) = muGCRF;
    sigma_final(:,:,Data.Ttr+step_ahead) = sigmaGCRF;

end

end