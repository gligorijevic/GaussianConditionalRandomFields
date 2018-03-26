function [values similarities R alphas] = temporalGraph3(T,N,historyLength)
%    alpha = 5;
%    beta = 0.25;
    alpha = 5;
    beta = 0.82;
    %beta = alpha*0.4522*exp(-0.05371 * commSize)/5;
    
    x = random('uniform', 0, 1, 3, N, T);
    alphas = random('uniform', -1, 1, 3, N);
    values=zeros(N,T);
    
    R = zeros(N, T);
    for t=1:T
        R(:,t) = sum(alphas.*x(:,:,t),1)';
    end
    clear x;
    b = 2*alpha*R;
    b = b(:);
    
    S = random('uniform', 0, N, N, N);
    for i=1:N
       S((i-1)+1:i, (i-1)+1:i) = random('uniform', 0, 1, 1, 1);
    end
    
    S = (S + S')/2;
    S = S - diag(diag(S));
    Q2 = zeros(N*T,N*T);
    
    betaS = beta*S;
    aux = sum(betaS,2);
    aux1 = diag(aux) - betaS;
    mu=zeros(1,N*T);
    v=zeros(N*T,1);
    for t=1:T
        Q2((t-1)*N+1:t*N, (t-1)*N+1:t*N) = aux1;
        sigma = inv(2*(aux1 + alpha*eye(N)));
        mu((t-1)*N+1:t*N)=sigma*b((t-1)*N+1:t*N);
        v((t-1)*N+1:t*N)=mvnrnd(mu((t-1)*N+1:t*N),sigma);
    end

%     Q1=alpha*eye(N*T);
%     sigmaInv = 2*(Q1+Q2);    
%     disp('start inv');
%     sigma = inv(sigmaInv);
%     disp('end inv');

    similarities = -Q2(1:N,1:N);
    similarities = similarities - diag(diag(similarities))+eye(N);
    
    for t=1:T
        values(:,t)=v((t-1)*N + 1:t*N);
    end
    
end
