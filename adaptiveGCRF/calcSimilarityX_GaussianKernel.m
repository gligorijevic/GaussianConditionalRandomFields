function [ GaussKernel ] = calcSimilarityX_GaussianKernel( thetas_Sl, Data ,nts )
% This function calculates the (different version) Gaussian kernel using
% parameters theta_Sl and input parameters thetas_Sl.
%
% thetas_Sl is in this case scalar
% similarity is calculated by formula:
% GausKernel(x1,x2)= exp(- (||x1-x2||)^2 / 2*sigma^2)

n = Data.N;

if Data.nthetas_Sl == 1
    GaussKernel = reshape(exp(-sum(Data.X_sim_dist_sq{nts},2) / 2*(thetas_Sl.^2)),n,n);
    return
else if Data.nthetas_Sl == numel(Data.xsim)
        % temp_sum is [ (x11-x21)^2/(2*theta1^2) (x12-x22)^2/(2*theta2^2); ...  ]
        temp_sum = bsxfun(@rdivide, Data.X_sim_dist_sq{nts} , 2*(thetas_Sl.^2));
        GaussKernel = reshape(exp(-sum(temp_sum,2)),n,n);
        return
    else if Data.nthetas_Sl == numel(Data.xsim) + 1
            temp_sum = bsxfun(@rdivide, Data.X_sim_dist_sq{nts} , 2*(thetas_Sl(2:end).^2));
            GaussKernel =  thetas_Sl(1)*reshape(exp(-sum(temp_sum,2)),n,n);
            return
        else if Data.nthetas_Sl == numel(Data.xsim) + 2
                temp_sum = bsxfun(@rdivide, Data.X_sim_dist_sq{nts} , 2*(thetas_Sl(3:end).^2));
                GaussKernel =  thetas_Sl(2)*reshape(exp(-sum(temp_sum,2)),n,n) + thetas_Sl(1)*eye(n);
                return
            end
        end
    end
end

end

