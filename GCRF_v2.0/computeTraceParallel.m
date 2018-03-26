function [ prodtrace ] = computeTraceParallel( sigma, dInverseSigma )
prodtrace = trace(sigma\dInverseSigma);
% MAXSIZE = 2000000;
% n = size(dInverseSigma,2);
% p = symrcm(sigma);
% % blockSize = floor(MAXSIZE/n);
% blockSize = ceil(n/16);
% % iter = floor(n/blockSize)+1;
% iter = 16;
% prodtraceTemp = zeros(iter,1);
% parfor k=1:iter % loop over columns of dInverseSigma
%     i = (k-1)*blockSize + 1;
%     toColumn = min([i+blockSize-1 n]);
%     x = sigma(p,p)\dInverseSigma(p,p(i:toColumn));
%     prodtraceTemp(k) = prodtraceTemp(k) + trace(x(i:toColumn,:));
% end
% prodtrace = sum(prodtraceTemp);