function [ d_theta_alpha ] = derivative_theta_alpha( mu, ylabel, CRFData, nts )

%x = [];
% % % for i = 1:CRFData.Ttr
% % %     x = [x; CRFData.x(:,CRFData.alpha_features,i)];
% % % end
% zamena za ovo gore je ova dole linija
x = reshape(ipermute(CRFData.x(:, CRFData.alpha_features, nts),[1 3 2]),[],length(CRFData.alpha_features));
x = [ones(size(x,1),1) ,x];

d_theta_alpha = nan(CRFData.N,(length(CRFData.alpha_features)+1));

% for i = 1:length(CRFData.predictors)
%     for j = 1:size(x,2)
%         asd = (i-1)*j+j %TODO sredi ovo iteriranje
        d_theta_alpha(:,:)= - x(:,:).*repmat((ylabel - mu), 1, length(CRFData.alpha_features)+1);
%     end
% end

end