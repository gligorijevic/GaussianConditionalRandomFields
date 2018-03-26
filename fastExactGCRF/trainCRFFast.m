function [ualpha, ubeta]= trainCRFFast(CRFData, useWorkers)
Data = CRFData;
ualpha = zeros(1, CRFData.noAlphas);
ubeta = zeros(1, CRFData.noBetas);

x0 = [ualpha ubeta]

c = parcluster('local'); % build the 'local' cluster object
noCores = c.NumWorkers; % get the number of workers
if strfind(version, '2015')
    if useWorkers == true
        poolobj = gcp('nocreate'); % If no pool, do not create new one.
        if isempty(poolobj)
            parpool('local',noCores);
        else
            poolsize = poolobj.NumWorkers;
        end
    end
else if  strfind(version, '2014')
        if useWorkers == true
            poolobj = gcp('nocreate'); % If no pool, do not create new one.
            if isempty(poolobj)
                parpool('local',noCores);
            else
                poolsize = poolobj.NumWorkers;
            end
        end
    else
        if useWorkers == true
            needNewWorkers = (matlabpool('size') == 0);
            if needNewWorkers
                % Open a new MATLAB pool with 4 workers.
                matlabpool open noCores
            end
        end
    end
end
options = optimset('Display','Iter','MaxIter',CRFData.maxiter,'GradObj','on'); %optimization
options = optimset(options,'UseParallel','always');

u = fminunc(@(x)objectiveCRFFast(x, CRFData),x0,options);
ualpha = u(1:CRFData.noAlphas);
ubeta = u(CRFData.noAlphas + 1:end);
