function [ ualpha, ubeta ] = trainCRF_new( CRFData, useWorkers )

ualpha = zeros(1, CRFData.noAlphas_new*CRFData.N);
ubeta = zeros(1, CRFData.noBetas_new);

x0 = [ualpha ubeta];

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

u = fminunc(@(x)objectiveCRF_new(x, CRFData),x0,options);
ualpha = u(1:CRFData.noAlphas_new*CRFData.N);
ubeta = u(CRFData.noAlphas_new*CRFData.N + 1:end);
end

