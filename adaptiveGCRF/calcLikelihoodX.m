function likelihood = calcLikelihoodX(RR, Q, ylabel, mu)
likelihood = - sum(log(diag(Q))) + 0.5*(ylabel-mu)'*Q*(ylabel-mu);
end