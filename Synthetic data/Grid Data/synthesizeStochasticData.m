function [ y, X, S, R, alpha, beta ] = synthesizeStochasticData(...
    T,N,alpha,beta,minSimValue, maxSimValue)
%SYNTHESIZESTOCHASTICDATA
% NOTE! you have to have even number of nodes, N%4=0

%% GENERATE R USING MACKAY-GLASS TIMESERIES MODELS
R=zeros(N, T);

a        = [0.0001 0.1 0.002 0.2 0.05 0.5 0.06 0.04 0.03 0.009];     % value for a in eq (1)
b        = [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.0001];     % value for b in eq (1)
tau      = [11 12 13 14 15 16 17 21 22 23];		% delay constant in eq (1)
x0       = [0.8 1.0 1.2 0.6 1.4 1.6 1.8 2.0 0.4 0.32];		% initial condition: x(t=0)=x0
deltat   = 0.1;	    % time step size (which coincides with the integration step)
sample_n = T-1; %12000;	% total no. of samples, excluding the given initial condition
interval = 1;	    % output is printed at every 'interval' time steps

% Main algorithm
% * x_t             : x at instant t         , i.e. x(t)        (current value of x)
% * x_t_minus_tau   : x at instant (t-tau)   , i.e. x(t-tau)
% * x_t_plus_deltat : x at instant (t+deltat), i.e. x(t+deltat) (next value of x)
% * X               : the (sample_n+1)-dimensional vector containing x0 plus all other computed values of x
% * T               : the (sample_n+1)-dimensional vector containing time samples
% * x_history       : a circular vector storing all computed samples within x(t-tau) and x(t)

for series = 1:N

time = 0;
index = 1;

tau_select = ceil(10*rand(1));

history_length = floor(tau(tau_select)/deltat);
x_history = zeros(history_length, 1); % here we assume x(t)=0 for -tau <= t < 0
x0_select = ceil(10*rand(1));
x_t = x0(x0_select);

X = zeros(sample_n+1, 1); % vector of all generated x samples
TT = zeros(sample_n+1, 1); % vector of time samples

for i = 1:sample_n+1,
    X(i) = x_t;
    if (mod(i-1, interval) == 0),
        disp(sprintf('%4d %f', (i-1)/interval, x_t));
    end
    if tau(tau_select) == 0,
        x_t_minus_tau = 0.0;
    else
        x_t_minus_tau = x_history(index);
    end
    
    a_select = ceil(10*rand(1));
    b_select = ceil(10*rand(1));
    x_t_plus_deltat = mackeyglass_rk4(x_t, x_t_minus_tau, deltat, a(a_select), b(b_select));
    
    if (tau(tau_select) ~= 0),
        x_history(index) = x_t_plus_deltat;
        index = mod(index, history_length)+1;
    end
    time = time + deltat;
    TT(i) = time;
    x_t = x_t_plus_deltat;
end

%     figure
%     plot(TT, X);
%     set(gca,'xlim',[0, TT(end)]);
%     xlabel('t');
%     ylabel('x(t)');
%     title(sprintf('A Mackey-Glass time serie (tau=%d)', tau));
    R(series,:) = X;
end
    
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

[sim] = generateGrid(N, minSimValue, maxSimValue); % argumenti su broj nodova, minimalna i maximalna vrednost sli?nosti
for t=1:T
    S(:,:,t)= sim;
end

%% Calculate S based on distance between data points (based on x)

%      S = zeros(N,N,T);
%     for t=1:T
%         dist = pdist(x(:,:,t));
%         sparse_indices = randperm(size(dist,2), floor(sparseness*size(dist,2)));
%         dist(sparse_indices) = Inf;
%         S(:,:,t) = exp(-squareform(dist)) - eye(N);
%     end
%    clearvars sparse_indices dist

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

X = NaN(T,N,1);

end