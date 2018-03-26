% function [ dSl_ThetaSl ] = s_derivativeSl_ThetaSl_GaussianKernel1(Data , thetas_Sl, nts)
% % derivative of the Gaussian Kernel (different versions)
% 
% if Data.nthetas_Sl == 1
%     [ GaussKernel, dist ] = calcSimilarityX_GaussianKernel1( thetas_Sl, Data ,nts);
%     dSl_ThetaSl = dist.^2/(thetas_Sl^3).*GaussKernel;
%     return
% else if Data.nthetas_Sl == numel(Data.xsim)
%         n =Data.N;
%         dSl_ThetaSl = NaN(n,n,Data.nthetas_Sl);
%         
%         xx = Data.x(:, Data.xsim, nts);
%         X = reshape(ipermute(xx,[1 3 2]), [], length(Data.xsim));
%         
%         xscaled = X./thetas_Sl;
%         [xx yy] = meshgrid(1:n,1:n);
%         temp_sum = ((xscaled(xx,:)-xscaled(yy,:)).^2)./2;
%         GaussKernel = reshape(exp(-sum(temp_sum,2)),n,n);
%         
%         for i = 1:Data.nthetas_Sl
%             dSl_ThetaSl(:,:,i) = GaussKernel.*reshape(sum(temp_sum,2),n,n)*(2/thetas_Sl(i)^2);
%         end
%         
%         return
%     else if Data.nthetas_Sl == numel(Data.xsim) + 1
%             n =Data.N;
%             dSl_ThetaSl = NaN(n,n,Data.nthetas_Sl);
%             
%             xx = Data.x(:, Data.xsim, nts);
%             X = reshape(ipermute(xx,[1 3 2]), [], length(Data.xsim));
%             
%             xscaled = X./thetas_Sl(2:end);
%             [xx yy] = meshgrid(1:n,1:n);
%             temp_sum = ((xscaled(xx,:)-xscaled(yy,:)).^2)./2;
%             GaussKernel =  thetas_Sl(1)*reshape(exp(-sum(temp_sum,2)),n,n);
%             
%             dSl_ThetaSl(:,:,1) = GaussKernel;
%             for i = 2:Data.nthetas_Sl
%                 dSl_ThetaSl(:,:,i) = thetas_Sl(1)*GaussKernel.*reshape(sum(temp_sum,2),n,n)*(2/thetas_Sl(i)^2);
%             end
%             
%             return
%         else if Data.nthetas_Sl == numel(Data.xsim) + 2
%                 n =Data.N;
%                 dSl_ThetaSl = NaN(n,n,Data.nthetas_Sl);
%                 
%                 xx = Data.x(:, Data.xsim, nts);
%                 X = reshape(ipermute(xx,[1 3 2]), [], length(Data.xsim));
%                 
%                 xscaled = X./thetas_Sl(3:end);
%                 [xx yy] = meshgrid(1:n,1:n);
%                 temp_sum = ((xscaled(xx,:)-xscaled(yy,:)).^2)./2;
%                 GaussKernel =  thetas_Sl(2)*reshape(exp(-sum(temp_sum,2)),n,n) + params(1)*eye(n);
%                 
%                 dSl_ThetaSl(:,:,1) = eye(n,n);
%                 dSl_ThetaSl(:,:,2) = GaussKernel;
%                 for i = 2:Data.nthetas_Sl
%                     dSl_ThetaSl(:,:,i) = thetas_Sl(2)*GaussKernel.*reshape(sum(temp_sum,2),n,n)*(2/thetas_Sl(i)^2);
%                 end
%                 
%                 return
%             end
%         end
%     end
% end
% 
% end
% 
