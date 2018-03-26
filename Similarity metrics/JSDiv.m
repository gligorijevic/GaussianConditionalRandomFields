function dist=JSDiv(P,Q)
%  dist = 1/2 * (KLDiv(P,M) + KLDiv(Q,M)) 
%  Jensen-Shannon divergence of two discrete probability distributions

M = (P + Q) .* 0.5;
dist = 0.5 .* (KLDiv(P,M) + KLDiv(Q,M));

end

