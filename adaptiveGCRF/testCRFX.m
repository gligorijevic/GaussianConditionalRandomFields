function [ predictionCRF, Sigma, Variance, predictionCRF_all, Sigma_all, Variance_all ]= testCRFX(Data)
nt = Data.N*Data.Ttr;
n = Data.N;
t = Data.Ttr;


%% split vector u to find ualpha and ubeta
ualpha = Data.u(1:Data.nalpha);
ubeta = Data.u(Data.nalpha+1:Data.nalpha+Data.nbeta);
thetas_Rk = Data.u(Data.nalpha+Data.nbeta+1:Data.nalpha+Data.nbeta+Data.nthetas_Rk);
thetas_Sl = Data.u(Data.nalpha+Data.nbeta+Data.nthetas_Rk+1:end);


allCovarianceMatrices = {};
fullMeanVector = [];

for nts = 1:Data.T
    
    Q1 = sparse(n, n);
    b = zeros(n, 1);
    
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
    end;
    
    Q2 = sparse(n,n);
    
    %     for i=1:Data.nbeta
    %         Q2 = Q2 - exp(ubeta(i)) * calcSimilarityX_GaussianKernel1(thetas_Sl, Data, nts);
    %     end;
    
    Sl_nts{1}=calcSimilarityX_GaussianKernel(thetas_Sl, Data, nts);
    Q2 = Q2 - exp(ubeta(1)) * Sl_nts{1};
    
    aux = sum(Q2,2);
    Q2diag = spconvert([(1:length(aux))', (1:length(aux))', aux]);
    Q2 = Q2 - Q2diag;
    
    Q = 2*(Q1 + Q2);
    
    RR= chol(Q);
    munew = RR\(RR'\b);
    
    Sigma = RR\(RR'\speye(n));
    
    allCovarianceMatrices{nts} = Sigma;
    fullMeanVector = [fullMeanVector; munew];
    
end

predictionCRF_all = fullMeanVector;
predictionCRF = fullMeanVector(end - Data.N * Data.Ttest + 1 : end);
Sigma_all = blkdiag(allCovarianceMatrices{:});
Variance_all = diag(Sigma_all);
Variance = diag(Sigma_all(end - Data.N * Data.Ttest + 1 : end, end - Data.N * Data.Ttest + 1 : end));
Sigma = Sigma_all(end - Data.N *Data.Ttest + 1 : end, end - Data.N *Data.Ttest + 1 : end);

end