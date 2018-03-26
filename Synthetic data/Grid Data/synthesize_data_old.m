function [y similarities R x alpha beta] = synthesize_data(numTimeSteps,T, N,alpha,beta,sparseness, no_features, no_hiddenNeurons,minSimilarityValue,maxSimilarityValue,xmin, xmax,noise_min, noise_max)


%% GENERATE temporal R with linear regression


x=zeros(N, no_features, numTimeSteps);
xi = random('uniform',(xmax-xmin)/2, xmax, N, no_features, 1); % uniform(0,1) -> Matrica (1(features),N,T)
%xunstr=zeros(N,no_features,1) ;

%% zbudz za unstructured  
xunstr=xi;



xtsInd=randperm(N);
xtsInd=xtsInd(1:round(0.8*N));
xts=xi(xtsInd,:,:);  
xts(:,mod(1,no_features)+1)=xts(:,mod(1,no_features)+1)*1.5;
xunstr(xtsInd,:,1)=xts;

% % % % 
% % % % 
% % % % for ts=2:size(xunstr,3)
% % % %     xunstr(:,:,ts)= xunstr(:,:,ts-1);
% % % %    
% % % %     xtsInd=randperm(N);
% % % %     xtsInd=xtsInd(1:round(0.5*N));
% % % %     
% % % %     xts=xunstr(xtsInd,:,ts-1);
% % % %     if mod(ts,2)==0
% % % %         xts(:,mod(ts,no_features)+1)=xts(:,mod(ts,no_features)+1)*1.1; %selektujemo samo jedno x da se menja u tom timestepu: mod(ts,no_features)+1
% % % %         
% % % %     else
% % % %         xts(:,mod(ts,no_features)+1)=xts(:,mod(ts,no_features)+1)*0.9; %selektujemo samo jedno x da se menja u tom timestepu: mod(ts,no_features)+1
% % % %     end
% % % %     xunstr(xtsInd,:,ts)=xts;
% % % % end
% % % % 




x(:,:,1)= xunstr(:,:,end);

for ts=2:T(2)
    x(:,:,ts)= x(:,:,ts-1);
   
    xtsInd=randperm(N);
    xtsInd=xtsInd(1:round(0.4*N));
    
    xts=x(xtsInd,:,ts-1);
    if mod(ts,2)==0
        xts(:,:)=xts*1.015; %svi xevi menjaju vrednost sem jednog ovog dole
        %xts(:,mod(ts,no_features)+1)=xts(:,mod(ts,no_features)+1)*1.015; %selektujemo samo jedno x da se menja u tom timestepu: mod(ts,no_features)+1
        
    else
        xts(:,:)=xts*1.01;
        %%xts(:,mod(ts,no_features)+1)=xts(:,mod(ts,no_features)+1)*1.01; %selektujemo samo jedno x da se menja u tom timestepu: mod(ts,no_features)+1
    end
    xts(:,mod(ts,no_features)+1)=x(xtsInd,mod(ts,no_features)+1,ts-1);
    x(xtsInd,:,ts)=xts;
end


for ts=T(2)+1:numTimeSteps
    x(:,:,ts)= x(:,:,ts-1);
   
    xtsInd=randperm(N);
    xtsInd=xtsInd(1:round(0.4*N));
    
    xts=x(xtsInd,:,ts-1);
    if mod(ts,2)==0
        xts(:,:)=xts*1.015;

        %%xts(:,mod(ts,no_features)+1)=xts(:,mod(ts,no_features)+1)*1.015; %selektujemo samo jedno x da se menja u tom timestepu: mod(ts,no_features)+1
        
    else
        xts(:,:)=xts*1.01;
        %%xts(:,mod(ts,no_features)+1)=xts(:,mod(ts,no_features)+1)*1.01; %selektujemo samo jedno x da se menja u tom timestepu: mod(ts,no_features)+1
    end
    xts(:,mod(ts,no_features)+1)=x(xtsInd,mod(ts,no_features)+1,ts-1);

    x(xtsInd,:,ts)=xts;
end





% % % % % % % %     x=sort(sort(x,2));
% % % % % % % %     theta = ones(no_features, N);  % separate theta for each node!
% % % % % % %     R = zeros(N, numTimeSteps);
% % % % % % %     theta = rand(no_features,no_hiddenNeurons); % za NN
% % % % % % %     hiddenTheta=rand(no_hiddenNeurons,1);  
% % % % % % % for t=1:numTimeSteps
% % % % % % %     %R(:,t) = x(:,:,t)*theta; %lin regr
% % % % % % %     sigm=logsig(x(:,:,t)*theta);% NN
% % % % % % %     R(:,t)=sigm*hiddenTheta;  % NN
% % % % % % % end


R = zeros(N, numTimeSteps);
for i=1:N
    
    thetai_lag0 = rand(no_features,1); % za LR sa lagovima 0
    mdl_lag0 = squeeze(x(i,:,1:end))'*thetai_lag0;

    thetai_lag1 = [thetai_lag0*0.8; rand(no_features,1)]; % za LR sa lagovima 1
    mdl_lag1 = [squeeze(x(i,:,1:end-1))', squeeze(x(i,:,2:end))'] *thetai_lag1;
    mdl_lag1=[mdl_lag0(1,:); mdl_lag1];
    
    
    thetai_lag2 = [thetai_lag1*0.8; rand(no_features,1)]; % za LR sa lagovima 2
    mdl_lag2 = [squeeze(x(i,:,1:end-2))', squeeze(x(i,:,2:end-1))', squeeze(x(i,:,3:end))']*thetai_lag2;
    mdl_lag2=[mdl_lag1(1:2,:); mdl_lag2];
    
    
    R(i,:)= 0.3*mdl_lag0 +0.4*mdl_lag1+ 0.3* mdl_lag2;
end



%% sortiranje x-a i R-a po gridu 
    i=1:N;
    i=int32(i);
    L=sqrt(N);
    L=int32(L);
    yax=idivide(i,L);
    yax=yax+1;
    xax=mod(i,L);
    yax(xax==0)=yax(xax==0)-1;
    xax(xax==0)=L;
    coordinate=[i' xax' yax' (xax' +yax')];
    sortcor=sortrows(coordinate,4);


    sortindexes=sortcor(:,1) ;

%      R(sortindexes,:)=blaj;
%      x(sortindexes,:,:)=xmed;


    [blaj, ind]=sort(R,1);
     xmed=zeros( N, no_features,numTimeSteps);
     for t=1:numTimeSteps
        xmed(:,:,t)=x(ind(:,t),:,t);
     end
     
     %x(ind,:,:)=xold;
     R(sortindexes,:)=blaj;
     x(sortindexes,:,:)=xmed;





% % % % % % 
    noise = random('norm', 0,0.1, N, numTimeSteps);%0.01
    R = R + noise;
% % % % % %     
% % %     
% % % % % %     
% % %       noise = random('uniform', noise_min, noise_max,N, no_features, numTimeSteps);%0.01 mean, std dev 
% % %       x=x+noise;
% % % % % %     



%     %% GENERATE S RANDOMLY
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
%%  generate s as chain
%     S = zeros(N,N,T);
%
%     for t=1:T
%        U=rand(N,N);
%        U=triu(U,1)-triu(U,2);
%        U=U+tril(U,1).';
%        S(:,:,t)= U;
%     end

%% generate S as grid
[sim] = generateGrid(N, minSimilarityValue, maxSimilarityValue); % argumenti su broj nodova, minimalna i maximalna vrednost sli?nosti
for t=1:numTimeSteps
    S(:,:,t)= sim;
end


%% calculate distance between data points based on x
%     for t=1:numTimeSteps
%         dist = pdist(x(:,:,t));
%         sparse_indices = randperm(size(dist,2), floor(sparseness*size(dist,2)));
%         dist(sparse_indices) = Inf;
%         S(:,:,t) = exp(-squareform(dist)) - eye(N);
%     end

%%


%% CALCULATE GCRF MATRICES

B = 2*alpha*R;

y = zeros(N,numTimeSteps);

for t = 1:numTimeSteps,
    
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

% figure;
% 
% subplot(3,2,1)
% scatter(xax, yax, [], R((yax-1)*L+xax,numTimeSteps),'filled') 
% colorbar;
% caxis([min(y(:,numTimeSteps)) max(y(:,numTimeSteps))])
% 
% title('R')
% 
% 
% subplot(3,2,2)
% scatter(xax, yax, [], y((yax-1)*L+xax,numTimeSteps),'filled')
% colorbar;
% caxis([min(y(:,numTimeSteps)) max(y(:,numTimeSteps))])
% title('y')

%% REPORT RESULTS
% sigma
% Q = inv(sigma)
% R_mu_y = [R(:,end),mu,y(:,end)]

%     S = S(:,:,1);
for i = 1: size(S, 3)
    similarities{i} = S(:,:,i);
end

clearvars -except  y similarities R x alpha beta theta 

end
