function [y, S, R, x, alpha, beta, theta] = ...
synthesize_data(T,N,alpha,beta,sparseness, no_features, no_hiddenNeurons, ...
xmin1, xmax1,xmin2, xmax2,noise_min, noise_max)

%%NOTE! you have to have even number of nodes, N%4=0


    %% GENERATE X

    %x for Similarity 1-  fisrt half of data comes from one distribution
    x_1 = random('uniform',xmin1, xmax1, N/2, no_features, T); % uniform(0,1) -> Matrica (1(features),N,T)
    
    %x for Similarity 2- second half of data comes from the other
    %distribution
    x_2 = random('uniform',xmin2, xmax2, N/2, no_features, T); % uniform(0,1) -> Matrica (1(features),N,T)

    %merge x_1 and x_2, first half is from first distribution and second
    %half is from second distribution
    x=[x_1;x_2];
    
    %x for Neural Network- take first half from first distribution and
    %second half from second distribution to generate R with neural
    %network, so it is 1. 1/4 and last 1/4 of x
    xNN = [x(1:N/4,:,:); x(3/4*N+1:end,:,:)];
    
    %x for Linear Regression - take second half from first distribution and
    %fist half from second distribution to generate R with neural network,
    %so it is 2. 1/4 and 3. 1/4 of x
    xLR = [x(N/4+1:3/4*N,:,:)];
    
    R=zeros(N,T);
    %% GENERATE R with Neural Network
    
    theta = rand(no_features,no_hiddenNeurons); % generate weights of input layer
    hiddenTheta=rand(no_hiddenNeurons,1);  % generate weights of hidden layer
    RNN = zeros(N/2, T); %this unstructured predictor has half of cases of x
    for t=1:T
        sigm=logsig(xNN(:,:,t)*theta); %logsig is transfer function for hidden layer
        RNN(:,t)=sigm*hiddenTheta;  % and output function is linear
    end
   R(1:N/4,:)= RNN(1:N/4,:); %this is to put on the first quarter of R predictor
   R(3/4*N+1:end,:)=RNN(N/4+1:end,:); %this is to put on the last quarter of R predictor
    
    %% GENERATE R with linear regression
    
    
    thetaLR = rand(no_features,1);  %generate weights for linear regression
    RLR = zeros(N/2, T); %this unstructured predictor has half of cases of x
    for t=1:T
      RLR(:,t) = xLR(:,:,t)*thetaLR; %linear regression

    end
    R(N/4+1:2*N/4,:)= RLR(1:N/4,:);%this is to put on the second quarter of R predictor
    R(2*N/4+1:3/4*N,:)=RLR(N/4+1:end,:);%this is to put on the third quarter of R predictor
         
     
    noise = random('uniform', noise_min, noise_max,N, T);  %0.01 mean, std dev 
    R=R+noise; % adding uniform random noise to the predictions

     clearvars RLR RNN sigm hiddenTheta thetaLR noise  xLR xNN x_1 x_2

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
%     clearvars -except N T alpha beta R S x theta minSimValue1 maxSimValue1


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
%     [sim] = generateGrid(N, minSimValue1, maxSimValue1); % argumenti su broj nodova, minimalna i maximalna vrednost sli?nosti
%     for t=1:T
%        S(:,:,t)= sim;
%     end
    
    
    %% Calculate S based on distance between data points (based on x)
     S = zeros(N,N,T);
    for t=1:T
        dist = pdist(x(:,:,t));
        sparse_indices = randperm(size(dist,2), floor(sparseness*size(dist,2)));
        dist(sparse_indices) = Inf;
        S(:,:,t) = exp(-squareform(dist)) - eye(N);
    end
   clearvars sparse_indices dist
   
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

end