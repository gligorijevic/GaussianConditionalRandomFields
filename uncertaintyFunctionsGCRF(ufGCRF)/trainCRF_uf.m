function [ u_theta_alpha, u_theta_beta ] = trainCRF_uf( CRFData, useWorkers )

u_theta_alpha = ones(1, (length(CRFData.alpha_features)+1)*CRFData.noAlphas_uf*CRFData.N); 
u_theta_beta = ones(1, CRFData.noBetas_uf);

CRFData.theta_alpha = exp(u_theta_alpha); %TODO
CRFData.theta_beta = exp(u_theta_beta);

x0 = [u_theta_alpha u_theta_beta];

% if useWorkers == true
%     needNewWorkers = (matlabpool('size') == 0);
%     if needNewWorkers
%         % Open a new MATLAB pool with 4 workers.
%         matlabpool open 4
%     end
% end
if useWorkers == true
    poolobj = gcp('nocreate'); % If no pool, do not create new one.
    if isempty(poolobj)
        parpool('local',4);
    else
        poolsize = poolobj.NumWorkers;
    end
end

options = optimset('Display','Iter','MaxIter',CRFData.maxiter,'GradObj','on'); %optimization
options = optimset(options,'UseParallel','always');

u = fminunc(@(objpar)objectiveCRF_uf(objpar, CRFData),x0,options);

u_theta_alpha = u(1:length(CRFData.theta_alpha));
u_theta_beta = u(length(CRFData.theta_alpha)+1:end);

end

