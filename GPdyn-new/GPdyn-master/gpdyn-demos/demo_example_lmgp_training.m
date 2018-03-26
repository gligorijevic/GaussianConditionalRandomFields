% demo_example_lmgp_training - demonstration script file for LMGP model identification - training part. 

%% Description
% Demo to present the how to train (=identify) the LMGP model.
% Note: 
% It can be used only with Gaussian covariance function and
% with white noise model (sum of covSEard and covNoise). 

% See Also
% demo_example_LMGP_data.m, demo_example_LMGP_simulation.m, trainLMGP.m,
% gpSD00.m

%
% Changelog:
%
% 16.2.2015, Martin Stepancic:
%		 	-added compatibility with gpml>=3.1
%

clear;
global flag_LM_data_ident
close all;


%%%%% off-equlibrium data 
mat_data=load('example_data_lmgp_oeq.mat');
ind_oeq = 5:2:length(mat_data.train_data.u); 

utrain = mat_data.train_data.u(ind_oeq); 
xtrain = mat_data.train_data.x(ind_oeq); 
ytrain = mat_data.train_data.y(ind_oeq); 

%%%%% identified local-model data
if(flag_LM_data_ident == 1)
    load example_data_lmgp_eq_ident   
else
     load example_data_lmgp_eq_anal
end 

Yeq = eq.Y; 
Ueq = eq.U; 
dfdy = eq.dfdy;
dfdu = eq.dfdu;


%***************************
% training data construction
% functional  
input = [Yeq Ueq; xtrain utrain]; 
target = [Yeq; ytrain]; 
targetvar = NaN*ones(size(target)); 
% derivative 

inputDer = [Yeq Ueq]; 
targetDer = [dfdy dfdu]; 

% LM variance 

for ii=1:size(inputDer,1)    
    if(flag_LM_data_ident == 1)
%         eq.lm{ii}.CovarianceMatrix
        derivevar(ii,:) = reshape(eq.lm{ii}.CovarianceMatrix,1,4);
    else
        derivevar(ii,:) = reshape(0.01*eye(2),1,4); 
    end 
end 


% covariance function: SE + white noise 
covfunc = {'covSum',{'covSEard','covNoise'}}; 

%************************************************************************
% training

% set the mean, covariance, inference method, and likelihood function. These
% are all unused and dummy, since the gpSD00 function has its own implementation.
inffunc= 'infExact';
meanfunc='meanZero';
covfunc= 'covSEard';
likfunc= 'likGauss';

%initialize hyperparameters:
hyp0.cov=ones(size(input,2)+1,1);
hyp0.lik=-1;

lag = 1; 

hyp=trainLMGP([],inffunc,meanfunc,covfunc,likfunc,input,target,inputDer, targetDer, derivevar);

% validation on ident data 
for ii=1:size(input,1)
    test = input(ii,:);
    [mug(ii) s2(ii)]=gpSD00(hyp,inffunc,meanfunc,covfunc,likfunc,input,target,targetvar, inputDer, targetDer, derivevar, test);
end

figure(299);
plot(Ueq,Yeq,'r',utrain(:,1),ytrain,'or',[input(:,2)],mug,'k*',...
    [input(:,2)],mug+2*s2,'k.',[input(:,2)],mug-2*s2,'k.');

disp('Hyperparameters: ');
disp(num2str(exp(unwrap(hyp))));
disp(' ');
disp(['Number of local models: ', num2str(length(targetDer))]);
disp(['Number of points out of equilibrium: ', num2str(length(utrain))]);
disp(['Number of training vectors: ', num2str(length(target)+numel(targetDer))]);
disp(' ');



if(1)
    save example_lmgp_trained hyp input target targetvar ...
        inputDer targetDer derivevar covfunc meanfunc inffunc likfunc lag;
end

return 





