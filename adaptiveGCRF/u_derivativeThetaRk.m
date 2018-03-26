function [ dP_ThetaRk ] = u_derivativeThetaRk( Data, ualpha, ylabel, mu, R_nts , nts)

     for k = 1:Data.nalpha
        dRk_thetaK=  u_derivativeRk_ThetaRk_LR( Data, ylabel, R_nts(:,k), nts );
        dP_ThetaRk = 2 * exp(ualpha(k))  * (ylabel - mu)' * dRk_thetaK;
     end

end
