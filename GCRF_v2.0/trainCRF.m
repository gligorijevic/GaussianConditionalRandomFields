function [ualpha, ubeta]= trainCRF(CRFData, useWorkers)

ualpha = zeros(1, CRFData.noAlphas*CRFData.N);
ubeta = zeros(1, CRFData.noBetas);

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

u = fminunc(@(x)objectiveCRF(x, CRFData),x0,options);

% u = lbfgs(@(x) objectiveCRF(x, CRFData), x0, 'Display', 'final');

ualpha = u(1:CRFData.noAlphas*CRFData.N);
ubeta = u(CRFData.noAlphas*CRFData.N + 1:end);
