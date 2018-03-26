function [mu, s2] = simulGPtaylorSE(hyp, inf, mean, cov, lik, input, target, test, lag)
% Simulation of the GP model with square exponential covariance function, where the 
% output variance is propagated using Taylor approximation.
%
%% Syntax
%  [mu, s2] = simulGPtaylorSE(hyp, inf, mean, cov, lik, input, target,
%  test, lag);
%
%% Description
% See A. Girard, Approximate Methods for Propagation of
% Uncertainty with Gaussian Process Models, PhD thesis, 2004.
% Simulation of the GP model, where the output variance is propagated using
% Taylor approximation. It can be used only with Gaussian 
% covariance function and with white noise model (covSEard and likGauss). 
% Uses routine gpTaylorSEard. 
% 
% Input: 
% * hyp      ... struct of optimized hyperparameters 
% * inf      ... function specifying the inference method 
% * mean     ... prior mean function
% * cov      ... specified covariance function, see help covFun for more info 
% * lik      ... likelihood function
% * input    ... input part of the training data,  NxD matrix
% * target   ... output part of the training data (ie. target), Nx1 vector 
% * test     ... input matrix for simulation, kxD vector, see
%                construct.m for more info 
% * lag      ... the order of the model (number of used lagged outputs) 
% 
% Output: 
% * mu       ... predictive mean when propagating the uncertainty 
% * s2       ... predictive variance when propagating the uncertainty  (including noise variance)
%
% See also:
% gpTaylorSEard, simulGPexactSE, simulGPnaive
%
% Examples:
% demo_example_gp_simulation.m
%
%% 
% Written by K. Azman, 2007, based on the software of J. Quinonero-Candela 
% and A. Girard. 

fun_name = 'simulGPtaylorSE'; 

x=input;
t=target;

[n D] = size(x);

% input validation
[ is_valid, hyp, inf, mean, cov, lik, msg ] = validate( hyp, inf, mean, cov, lik, D);

if ~isequal(cov,{@covSEard}) 
    error(strcat([fun_name,': function can only be called with the', ...
        ' covariance function ''covSEard'' '])); 
end

if ~isequal(lik,{@likGauss}) 
    error(strcat([fun_name,': function can only be called with the', ...
        ' likelihood function ''likGauss'', where hyp.lik parameter is log(sn)'])); 
end

X=[-2*hyp.cov(1:end-1);2*hyp.cov(end);2*hyp.lik]; % adapt hyperparameters to local format
expX = exp(X);
vy = expX(end);

SigmaX = zeros(D,D);
% training covariance matrix
Q = zeros(n,n);
for d = 1:D
  Q = Q + (repmat(input(:,d),1,n)-repmat(input(:,d)',n,1)).^2*expX(d);
end
Q = expX(D+1)*exp(-0.5*Q);

Q = Q + vy*eye(n);  % data cov. matrix: add noise model

invQ = inv(Q);

% 1st point
xtest = test(1,:); 

[mu(1), s2(1),deriv, S2deriv] = gpTaylorSEard(hyp, inf, mean, cov, lik, invQ, x, t, xtest, SigmaX);

s2(1)=s2(1);

for i=2:size(test,1)

   xtest = [xtest(2:lag) mu(i-1) test(i,lag+1:end)];

   covXY = deriv*SigmaX;

   SigmaX(1:lag-1,1:lag-1) = SigmaX(2:lag,2:lag);
   SigmaX(lag,lag) = s2(i-1);
   SigmaX(1:lag-1,lag) = covXY(2:lag)';
   SigmaX(lag,1:lag-1) = covXY(2:lag);

   [mu(i), s2(i),deriv, S2deriv] = gpTaylorSEard(hyp, inf, mean, cov, lik, invQ, x, t, xtest, SigmaX);
    s2(i)=s2(i);
end

% transform into coloumn vectors 
mu = mu'; 
s2 = s2'; 

return; 
