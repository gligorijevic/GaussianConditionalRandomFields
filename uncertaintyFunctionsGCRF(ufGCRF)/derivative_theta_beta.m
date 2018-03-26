function [ d_theta_beta ] = derivative_theta_beta( mu, ylabel, CRFData )

% x = reshape(CRFData.x, sum(CRFData.label)*CRFData.Ttr, size(CRFData.x,2));
% x = [];
% for i = 1:CRFData.Ttr
%     x = [x; CRFData.x(:,CRFData.beta_features,i)];
% end

x = reshape(ipermute(CRFData.x(:,CRFData.beta_features,1:CRFData.Ttr),[1 3 2]),[],length(CRFData.beta_features));

d_theta_beta = nan(CRFData.noBetas, length(CRFData.beta_features));

for i = 1:length(CRFData.similarities)
    for j = size(x,2)
        [xx yy] = meshgrid(1:CRFData.N*CRFData.Ttr,1:CRFData.N*CRFData.Ttr);
        temp_sum = (x(xx,:)-x(yy,:));
        
        temp_residuals = repmat((ylabel - mu), CRFData.N*CRFData.Ttr,1);
        (i-1)+j %TODO sredi ubacivanje vrednosti
        %d_theta_beta((i-1)+j) = - sum( temp_sum(:,j).*temp_residuals);
        d_theta_beta(i,j) = - sum( temp_sum(:,j).*temp_residuals);
    end
end
end