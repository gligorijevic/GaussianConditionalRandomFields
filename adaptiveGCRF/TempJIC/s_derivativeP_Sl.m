% function [dp_Sl] = s_derivativeP_Sl( Data,Q1,Q2,ylabel,mu,ubeta,thetas_Sl, nts )
% n = Data.N;
% dp_Sl = nan(Data.nbeta,Data.nthetas_Sl);
% 
% Q = 2*(Q1+Q2);
% Qll = Q(Data.label(n*(nts-1)+1 : n*nts),Data.label(n*(nts-1)+1 : n*nts));
% 
% for i=1:Data.nbeta
%     % matrix with -2*beta in each cell
%     dQ = -2*exp(ubeta(i))*ones(n,n);
%     
%     % sum elements in each row
%     aux = sum(dQ,2);
%     dQdiag = spconvert([(1:length(aux))', (1:length(aux))', aux]);
%     dQ = dQ - dQdiag;
%     dQll = dQ(Data.label(n*(nts-1)+1 : n*nts),Data.label(n*(nts-1)+1 : n*nts));
%     for dim = 1:Data.nthetas_Sl
%         dp_Sl(i, dim) = ...
%             (-0.5*(ylabel+mu)'*dQll.*s_derivativeSl_ThetaSl_GaussianKernel(Data, thetas_Sl, dim, nts)*(ylabel-mu) + ...
%             0.5*trace(Qll\(dQll.*s_derivativeSl_ThetaSl_GaussianKernel(Data , thetas_Sl,nts))))*exp(ubeta(i));
%     end
% end