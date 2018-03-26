function [u, ualpha, ubeta, thetas_Rk, thetas_Sl, pred, Q]= trainCRFX(Data)

ualpha = Data.thetaAlpha; %0.5*log(1000); % 1*ones(1,Data.nalpha);
ubeta = Data.thetaBeta; %0.5* log(5000); %1*ones(1,Data.nbeta);
thetas_Rk = Data.thetaR; % 1*ones(1, Data.nthetas_Rk);%
thetas_Sl = Data.thetaS; % 1*ones(1, Data.nthetas_Sl);

x0 = [ualpha ubeta thetas_Rk thetas_Sl];

c = parcluster('local'); % build the 'local' cluster object
noCores = c.NumWorkers; % get the number of workers
if strfind(version, '2015')
    if Data.useWorkers == true
        poolobj = gcp('nocreate'); % If no pool, do not create new one.
        if isempty(poolobj)
            parpool('local',noCores);
        else
            poolsize = poolobj.NumWorkers;
        end
    end
else if  strfind(version, '2014')
        if Data.useWorkers == true
            poolobj = gcp('nocreate'); % If no pool, do not create new one.
            if isempty(poolobj)
                parpool('local',noCores);
            else
                poolsize = poolobj.NumWorkers;
            end
        end
    else
        if Data.useWorkers == true
            needNewWorkers = (matlabpool('size') == 0);
            if needNewWorkers
                % Open a new MATLAB pool with 4 workers.
                matlabpool open noCores
            end
        end
    end
end
options = optimset('Display','Iter','GradObj','on','MaxIter',Data.maxiter);
options = optimset(options,'UseParallel','always');

u = fminunc(@(x)objectiveCRFX(x,Data),x0,options);
[~,~,pred,Q] = objectiveCRFX(u,Data);

Data.u = u;

ualpha = u(1:Data.nalpha);
ubeta = u(Data.nalpha+1:Data.nalpha+Data.nbeta);
thetas_Rk = u(Data.nalpha+Data.nbeta+1:Data.nalpha+Data.nbeta+Data.nthetas_Rk);
thetas_Sl = u(Data.nalpha+Data.nbeta+Data.nthetas_Rk+1:end);

