% function [ dSl_ThetaSl ] = s_derivativeSl_ThetaSl_EDScale(Data , thetaSl)
%     sim = exp( -Data.Dproj / thetaSl);
%     sim(Data.Dproj > 200) = 0;
%     sim(Data.Dproj == 0) = 0;
% 
%     Data.sim = sim;
%     dSl_ThetaSl = -1/(thetaSl*thetaSl)*mean(mean(Data.Dproj.*sim)); % thetaSL is a scalar
% end
% 
