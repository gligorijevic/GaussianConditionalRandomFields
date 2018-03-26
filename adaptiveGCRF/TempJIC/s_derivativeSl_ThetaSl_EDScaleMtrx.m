% function [ dSl_ThetaSl ] = s_derivativeSl_ThetaSl_EDScale( Data , thetaSl)
% %S_DERIVATIVESL_THETASL Summary of this function goes here
% %   Detailed explanation goes here
%     sim = exp(-Data.Dproj./thetaSl);
%     sim(Data.Dproj>200)=0;
%     sim(Data.Dproj==0)=0;
% 
%     Data.sim=sim;
%     dSl_ThetaSl=-(1./(thetaSl.^2)).*Data.Dproj.*sim; % NxN
% end
% 
% 
