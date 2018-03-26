function [y S R x alpha beta theta xmed xold] = synthesize_data(T,N,alpha,beta,sparseness, no_features, no_hiddenNeurons,minSimilarityValue,maxSimilarityValue,xmin, xmax,noise_min, noise_max)


    %% GENERATE R with linear regression
    i=1:N;
    i=int32(i);
    L=sqrt(N);
    L=int32(L);
    yax=idivide(i,L);
    yax=yax+1;
    xax=mod(i,L);
    % proverastaro=[i' xax' yax'];
    yax(xax==0)=yax(xax==0)-1;
    xax(xax==0)=L;
    coordinate=[i' xax' yax' (xax' +yax')];
    sortcor=sortrows(coordinate,4);

    xold = random('uniform',xmin, xmax, N, no_features, T); % uniform(0,1) -> Matrica (1(features),N,T)

    %%%%%%%%%%%% blaj=sort(sort(x,1),2);
    sortindexes=sortcor(:,1) ;
    %%%%%%%%%%%% x=blaj(sortindexes,:,:);

    %theta = ones(no_features, N);  % separate theta for each node!
    %theta = rand(no_features,1);  %za lin regr

    %theta=ones(no_features,1);   
   %%%%%%%%%%%% theta = sort(sort(diff([zeros(1,no_hiddenNeurons);sort(rand(no_features-1,no_hiddenNeurons));ones(1,no_hiddenNeurons)]),2),1);
    theta = rand(no_features,no_hiddenNeurons); % za NN

    %%%%%%%%%%%% hiddenTheta=sort(rand(no_hiddenNeurons,1),1)  ;
    hiddenTheta=rand(no_hiddenNeurons,1);  
    R1 = zeros(N, T);
    for t=1:T
      %  R(:,t) = x(:,:,t)*theta; %lin regr
        sigm=logsig(xold(:,:,t)*theta);
        R1(:,t)=sigm*hiddenTheta;  % NN
    end
     [blaj, ind]=sort(R1,1);
     xmed=zeros( N, no_features, T);
     for t=1:T
        xmed(:,:,t)=xold(ind(:,t),:,t);
     end
     %x(ind,:,:)=xold;
     R(sortindexes,:)=blaj;
     x(sortindexes,:,:)=xmed; 
% %     noise = random('uniform', noise_min, noise_max, N, T);%0.01
% %     R = R + noise;
      noise = random('uniform', noise_min, noise_max,N, no_features, T);%0.01 mean, std dev 
      x=x+noise;

%     clearvars -except N T alpha beta R x sparseness theta minSimilarityValue maxSimilarityValue no_hiddenNeurons 

     %% GENERATE S RANDOMLY
% 
%     S = 0+rand(N)*1;   % uniform random
%     upper_matrix_indices = find(triu(S,1)>0);   % get indices for matrix elements above diagonal 
%     sparse = randperm((N*N-N)/2, floor(sparseness*(N*N-N)/2));  % select random indices for upper diagonal
%     S(upper_matrix_indices(sparse)) = 0;  %  sparsify selected elements above diagonal
%     S = triu(S,1)+triu(S,1)';  % make symmetric matrix
%     
%     S = repmat(S,[1,1,T]);
%     
%     clearvars -except N T alpha beta R S x theta minSimilarityValue maxSimilarityValue


    %%  GENERATE S BASED ON X
  %% generate s as chain
%     S = zeros(N,N,T);
%     
%     for t=1:T
%        U=rand(N,N);
%        U=triu(U,1)-triu(U,2);
%        U=U+tril(U,1).';
%        S(:,:,t)= U;
%     end
    
 %% generate S as grid
%     [sim] = generateGrid(N, minSimilarityValue, maxSimilarityValue); % argumenti su broj nodova, minimalna i maximalna vrednost sli?nosti
%     for t=1:T
%        S(:,:,t)= sim;
%     end
    
    
    %% calculate distance between data points based on x
    for t=1:T
        dist = pdist(x(:,:,t));
        sparse_indices = randperm(size(dist,2), floor(sparseness*size(dist,2)));
        dist(sparse_indices) = Inf;
        S(:,:,t) = exp(-squareform(dist)) - eye(N);
    end

%%

   % clearvars -except N T alpha beta R S x sparseness theta minSimilarityValue maxSimilarityValue no_hiddenNeurons coordinate

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
    sigma =Q\eye(N);
    % b imam u Bt
    b = B(:,t);
    % izracunaj Mu(t)  [N]
    mu = sigma* b;
    y(:,t) = mvnrnd(mu,sigma);  %Create sample out of distribution
    
end

%% REPORT RESULTS
% sigma
% Q = inv(sigma)
% R_mu_y = [R(:,end),mu,y(:,end)]

    S = S(:,:,1);
end