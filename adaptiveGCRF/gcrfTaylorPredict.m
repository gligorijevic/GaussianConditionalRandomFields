function [ mu, sigma, derivMu, derivSigma, correctionTerm ] =...
    gcrfTaylorPredict( Data, SigmaX, correctionTerm, nts )

ualpha = Data.u(1:Data.nalpha);
ubeta = Data.u(Data.nalpha+1:Data.nalpha+Data.nbeta);
thetas_Rk = Data.u(Data.nalpha+Data.nbeta+1:Data.nalpha+Data.nbeta+Data.nthetas_Rk);
thetas_Sl = Data.u(Data.nalpha+Data.nbeta+Data.nthetas_Rk+1:end);

n = Data.N;
dim = size(Data.X_all{1},2);

% Obtain mu and Sigma from the GCRF
Q1 = sparse(n, n);
b = zeros(n, 1);

X = Data.X_unstr{nts};
R_nts=zeros(n, Data.nalpha);

for i=1:Data.nalpha
    R_nts(:,i) = X * thetas_Rk';
    bb =  2 * exp(ualpha(i)) * R_nts(:,i);
    b = b + bb(:);
    
    Q1 = Q1 + exp(ualpha(i)) * speye(n,n);
end;

Q2 = sparse(n,n);

Sl_nts=calcSimilarityX_GaussianKernel(thetas_Sl, Data, nts);
Q2 = Q2 - exp(ubeta(1)) * Sl_nts;

aux = sum(Q2,2);
Q2diag = spconvert([(1:length(aux))', (1:length(aux))', aux]);
Q2 = Q2 - Q2diag;

Q = 2*(Q1 + Q2);

RR= chol(Q);
munew = RR\(RR'\b);

Sigma = RR\(RR'\speye(n));

% Calculate correction term for the variance
[xx, yy] = meshgrid(1:Data.N,1:Data.N);

% for d = 1:dim
%     derivLR = 2.*exp(ualpha(i)).*thetas_Rk(d+1); % first derivative of the
%     % b because we optimize single unstructured predictor for all nodes
%     % if we would have a specific unstrucutred predictor for each node we
%     % would not have to do the repmat.
%     derivLR = repmat(derivLR, n, 1);
%
%     % The derivative part of the Gaussian kernel
%     a = (Data.X_sim{nts}(xx,d)-Data.X_sim{nts}(yy,d))/thetas_Sl(d);
%     a = reshape(a,n,n);
%
%     derivGK = Sl_nts{1}.*a; %first partial derivative of Q(of each element)
%
%     derivQ2 = sparse(n,n);
%     derivQ2 = derivQ2 - exp(ubeta(1)) * derivGK;
%
%     aux = sum(derivQ2,2);
%     derivQ2diag = spconvert([(1:length(aux))', (1:length(aux))', aux]);
%     derivQ2 = derivQ2 - derivQ2diag; %first partial derivative of Q (full)
%
%     % First partial derivative of the mean (full)
%     derivMu(:,d) = Sigma*derivLR + Sigma*derivQ2*Sigma*b;
% end

derivMu = NaN(n,dim);
derivSigma = {};
for d = 1:dim
    % Calculate first derivative of the mean for each dimension
    derivLR = 2.*exp(ualpha(i)).*thetas_Rk(d+1); % first derivative of the
    % b because we optimize single unstructured predictor for all nodes
    % if we would have a specific unstrucutred predictor for each node we
    % would not have to do the repmat.
    derivLR = repmat(derivLR, n, 1);
    
    % The derivative part of the Gaussian kernel
    aux1 = (Data.X_sim{nts}(xx,d)-Data.X_sim{nts}(yy,d))/thetas_Sl(d);
    aux1 = reshape(aux1,n,n);
    
    derivGK = Sl_nts.*aux1; %first partial derivative of Q(of each element)
    
    derivQ2 = sparse(n,n);
    derivQ2 = derivQ2 - exp(ubeta(1)) * derivGK;
    
    aux = sum(derivQ2,2);
    derivQ2diag = spconvert([(1:length(aux))', (1:length(aux))', aux]);
    derivQ2 = derivQ2 - derivQ2diag; %first partial derivative of Q (full)
    derivQ2 = 2*derivQ2;
    
    % First partial derivative of the mean (full)
    derivMu(:,d) = -Sigma*derivQ2*Sigma*b + Sigma*derivLR ;
    
    derivSigma{d} = {};
    for e = 1:dim
        % First partial derivative of Q w.r.t dimension d
        aux2 = (Data.X_sim{nts}(xx,d)-Data.X_sim{nts}(yy,d))/thetas_Sl(1+d);
        aux2 = reshape(aux2,n,n);
        
        derivGK_a = Sl_nts.*aux2; % dot product
        
        derivQ2_d = sparse(n,n);
        derivQ2_d = derivQ2_d - exp(ubeta(1)) * derivGK_a;
        
        aux = sum(derivQ2_d,2);
        derivQ2diag = spconvert([(1:length(aux))', (1:length(aux))', aux]);
        derivQ2_d = derivQ2_d - derivQ2diag; % first partial derivative of
        % the Q with respect to d'th dimension
        derivQ2_d = 2*derivQ2_d;
        
        % First partial derivative of Q w.r.t dimension e
        aux3 = (Data.X_sim{nts}(xx,e)-Data.X_sim{nts}(yy,e))/thetas_Sl(1+e);
        aux3 = reshape(aux3,n,n);
        
        derivGK_b = Sl_nts.*aux3; % dot product
        
        derivQ2_e = sparse(n,n);
        derivQ2_e = derivQ2_e - exp(ubeta(1)) * derivGK_b;
        
        aux = sum(derivQ2_e,2);
        derivQ2diag = spconvert([(1:length(aux))', (1:length(aux))', aux]);
        derivQ2_e = derivQ2_e - derivQ2diag; % first partial derivative of
        % the Q with respect to e'th dimension
        derivQ2_e = 2*derivQ2_e;
        
        % Second partial derivative of Q w.r.t dimension d and e
        %         aux4 = (Data.X_sim{nts}(xx,d)-Data.X_sim{nts}(yy,d))/thetas_Sl(1+d);
        %         aux4 = reshape(aux4,n,n);
        %         aux5 = (Data.X_sim{nts}(xx,e)-Data.X_sim{nts}(yy,e))/thetas_Sl(1+e);
        %         aux5 = reshape(aux5,n,n);
        derivGK_de = Sl_nts.*aux3.*aux2; % dot product
        
        derivQ2_de = sparse(n,n);
        derivQ2_de = derivQ2_de - exp(ubeta(1)) * derivGK_de;
        
        aux = sum(derivQ2_de,2);
        derivQ2diag = spconvert([(1:length(aux))', (1:length(aux))', aux]);
        derivQ2_de = derivQ2_de - derivQ2diag;
        derivQ2_de = 2*derivQ2_de; % Pomnoziti -1 ovde ako treba
        
        derivSigma{d}{e} = ...
            Sigma*(2*derivQ2_d*Sigma*derivQ2_e-derivQ2_de)*Sigma;
        
%         check = sum(sum(...
%             (Sigma*derivQ2_d*Sigma*derivQ2_e*Sigma) -...
%             (Sigma*derivQ2_e*Sigma*derivQ2_d*Sigma)))
        
    end
end

mu = munew;
sigma = Sigma;

% correctionTerm = 0;
b_part = NaN(n,1);
a_part = NaN(n,1);

for i=1:n
    b_part(i) = (derivMu(i,:)*SigmaX{i}*derivMu(i,:)');
end

for j = 1:n
    deriv_d = zeros(dim, dim);
    for d = 1:dim
        for e = 1:dim
            deriv_d(d,e) = derivSigma{d}{e}(j,j);
        end
    end
    a_part(j) = 0.5*(trace(deriv_d*SigmaX{j}));
end

% correctionTerm = diag(a_part + b_part);
correctionTerm = diag(abs(a_part) + abs(b_part));
% correctionTerm(correctionTerm < 0) = 0;
sigma = Sigma + correctionTerm;

end