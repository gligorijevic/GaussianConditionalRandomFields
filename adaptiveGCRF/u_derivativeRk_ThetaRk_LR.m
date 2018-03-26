function [ dRk_thetaK ] = u_derivativeRk_ThetaRk_LR( Data, ylabel, R_nts, nts )
% This is when we are optimising theta_Rk using MSE
%     Rk_error = 2*(ylabel - R_nts); %NxM
%     dTheta=-Data.X_unstr{nts};
%     dRk_thetaK = bsxfun(@times, Rk_error, dTheta); % dTheta_i= (y-R)* (-Xi)

% This is optimising theta_Rk
dTheta= Data.X_unstr{nts};
dRk_thetaK = dTheta;
end