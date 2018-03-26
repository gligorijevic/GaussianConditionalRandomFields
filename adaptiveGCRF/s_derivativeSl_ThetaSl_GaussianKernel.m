function [ dSl_ThetaSl ] = s_derivativeSl_ThetaSl_GaussianKernel(Data, Sl_nts,thetas_Sl, dim, nts)
% derivative of the Gaussian Kernel (different versions)
n =Data.N;
if Data.nthetas_Sl == 1
    dSl_ThetaSl = Sl_nts .* reshape(Data.X_sim_dist_sq{nts},n,n) * (1/thetas_Sl^3);
    return
else if Data.nthetas_Sl == numel(Data.xsim)
        dSl_ThetaSl = Sl_nts .* reshape(Data.X_sim_dist_sq{nts}(:,dim),n,n) * (1/thetas_Sl(dim)^3);
        return
    else if Data.nthetas_Sl == numel(Data.xsim) + 1
            if dim ==1
                dSl_ThetaSl = Sl_nts;
            else
                dSl_ThetaSl = Sl_nts .* reshape(Data.X_sim_dist_sq{nts}(:,dim-1),n,n) * (1/thetas_Sl(dim)^3);
            end
            return
        else if Data.nthetas_Sl == numel(Data.xsim) + 2
                if dim ==1
                    dSl_ThetaSl = 1;
                else if dim == 2
                        dSl_ThetaSl = Sl_nts;
                    else
                        dSl_ThetaSl = Sl_nts .* reshape(Data.X_sim_dist_sq{nts}(:,dim-2),n,n) * (1/thetas_Sl(dim)^3);
                    end
                end
                return
            end
        end
    end
end

end

