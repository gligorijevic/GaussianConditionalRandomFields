function dist=KLDist(P,Q, prior)
%  dist = 1/2 * (KLDiv(P,Q) + KLDiv(Q,P)) 
%  Symetric Kullback-Leibler divergence of two discrete probability distributions

if nargin < 3
    prior = 0;
end;

dist = 0.5 * (KLDiv(P,Q, prior) + KLDiv(Q,P, prior));

end

