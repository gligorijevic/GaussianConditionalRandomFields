function [ dThetaSl ] = s_derivativeThetaSl(Data, Qll,  Sl_nts , ylabel, mu, ubeta, thetas_Sl, nts)
n = Data.N;
dp_Sl = nan(Data.nbeta,Data.nthetas_Sl);

for i=1:Data.nbeta
    % matrix with -2*beta in each cell
    dQ = -2*exp(ubeta(i))*ones(n,n);
    
    % sum elements in each row
    aux = sum(dQ,2);
    dQdiag = spconvert([(1:length(aux))', (1:length(aux))', aux]);
    dQ = dQ - dQdiag;
    dQll = dQ(Data.label(n*(nts-1)+1 : n*nts),Data.label(n*(nts-1)+1 : n*nts));
    
    for dim = 1:Data.nthetas_Sl
        
        dSl_thSl_dim=dQll.*s_derivativeSl_ThetaSl_GaussianKernel(Data, Sl_nts{1}, thetas_Sl, dim, nts);
        dp_Sl(i, dim) = (-0.5*(ylabel+mu)'*dSl_thSl_dim*(ylabel-mu) + ...
            0.5*trace(dSl_thSl_dim/Qll));
    end


end
dThetaSl=[];
for j= 1:Data.nbeta
  dThetaSl= [dThetaSl, dp_Sl(j,:)];  
end

