function [x, y, S] = temporalGraph4(T,N,no_features,alpha,beta,sparseness)
%     clear;
%     clc;
%     N = 5;
%     T = 5;
    
    rng(10);  % set random seed
%     sparseness = 0.7;
    
%     alpha = 1;
%     beta = 0.1;
    
    
    %% GENERATE R
%     no_features = 5;
    x = random('uniform', 0.01, 0.1, N, no_features, T); % uniform(0,1) -> Matrica (1(features),N,T)
    
%     theta = ones(no_features, N);  % separate theta for each node!
    theta = random('uniform', 0.01,0.02,no_features,1);
    
%     x(:,:,6) = random('uniform', 30, 40, N, no_features);

    R = zeros(N, T);
    for t=1:T
        R(:,t) = x(:,:,t)*random('uniform', 0,2,no_features,1);
    end
    R(:,6) = x(:,:,6)*theta;
    
    noise = random('normal', 0, 0.1, N, T);
    R = R + noise;


    %% GENERATE R UNCERTAINTY
    no_features_unc = no_features;
    x_unc = random('uniform', 3, 4, N, no_features_unc, 1); % uniform(0,1) -> Matrica (1(features),N,T)
    
%     theta = ones(no_features, N);  % separate theta for each node!
%     theta = rand(no_features,1);
    R_unc = zeros(N, T);
    R_unc = R;
    
    R_unc(:,end) = x_unc(:,:,end)*theta + random('normal', 1, 1.5, N,1);
    
    x_unc = [reshape(ipermute(x(:,:,1:T-1),[1 3 2]),[], no_features); x_unc];
    
%     for t=1:T
%         R_unc(:,t) = x(:,:,t)*theta;
%     end
%     
 

%     %% GENERATE S RANDOMLY
% 
%     S = 0+rand(N)*1;   % uniform random
%     upper_matrix_indices = find(triu(S,1)>0);   % get indices for matrix elements above diagonal 
%     sparse = randperm((N*N-N)/2, floor(sparseness*(N*N-N)/2));  % select random indices for upper diagonal
%     S(upper_matrix_indices(sparse)) = 0;  %  sparsify selected elements above diagonal
%     S = triu(S,1)+triu(S,1)';  % make symmetric matrix
%     
%     S = repmat(S+eye(N,N),[1,1,T]);
    

%     %% GENERATE S RANDOMLY UNCERTAINTY
% 
%     S_unc = 0+rand(N)*1;   % uniform random
%     upper_matrix_indices_unc = find(triu(S_unc,1)>0);   % get indices for matrix elements above diagonal 
%     sparse_unc = randperm((N*N-N)/2, floor(sparseness*(N*N-N)/2));  % select random indices for upper diagonal
%     S_unc(upper_matrix_indices_unc(sparse_unc)) = 0;  %  sparsify selected elements above diagonal
%     S_unc = triu(S_unc,1)+triu(S_unc,1)';  % make symmetric matrix
%     
%     S_unc = repmat(S_unc,[1,1,T]);
%     


    %% GENERATE S BASED ON X

    
    S = zeros(N,N,T);
        
    for t=1:T
        dist = pdist(x(:,:,t));
        sparse_indices = randperm(size(dist,2), floor(sparseness*size(dist,2)));
        dist(sparse_indices) = Inf;
        S(:,:,t) = exp(-squareform(dist)) - eye(N);
    end
    
    
    
        %% GENERATE S unc BASED ON X

    
   S_unc = zeros(N,N,T);
        
    for t=1:T
        dist = pdist(x(:,:,t));
        sparse_indices = randperm(size(dist,2), floor(sparseness*size(dist,2)));
        dist(sparse_indices) = Inf;
        S_unc(:,:,t) = exp(-squareform(dist)) - eye(N);
    end

   % clearvars -except N T alpha beta R S x sparseness theta

    %% CALCULATE GCRF MATRICES
    
    B = 2*alpha*R;

    y = zeros(N,T);

for t = 1:T,

   % izracunaj Q na osnovu beta,alfa

    Q1 = alpha*eye(N);
    betaS = beta*S(:,:,t);
    Q2 = diag(sum(betaS,2)) - betaS;
    Q = 2 * (Q1 + Q2);
    
    % izracunaj sigma
    sigma = inv(Q);
    
    % b imam u Bt
    b = B(:,t);
    
    % izracunaj Mu(t)  [N]
    mu = sigma * b;
    
    y(:,t) = mvnrnd(mu,sigma);  %Create sample out of distribution
    
end

    %% CALCULATE GCRF MATRICES UNCERTAINTY
    
    B_unc = 2*alpha*R_unc;

    y_unc = zeros(N,T);

for t = 1:T,

   % izracunaj Q na osnovu beta,alfa

    Q1_unc = alpha*eye(N);
    betaS_unc = beta*S_unc(:,:,t);
    Q2_unc = diag(sum(betaS_unc,2)) - betaS_unc;
    Q_unc = 2 * (Q1_unc + Q2_unc);
    
    % izracunaj sigma
    sigma_unc = inv(Q_unc);
    
    % b imam u Bt
    b_unc = B_unc(:,t);
    
    % izracunaj Mu(t)  [N]
    mu_unc = sigma_unc * b_unc;
    
    y_unc(:,t) = mvnrnd(mu_unc,sigma_unc);  %Create sample out of distribution
    
end
%% REPORT RESULTS
% sigma
% Q = inv(sigma)
% R_mu_y = [R(:,end),mu,y(:,end)]

    S = S(:,:,1);
    S_unc = S_unc(:,:,1);
end
