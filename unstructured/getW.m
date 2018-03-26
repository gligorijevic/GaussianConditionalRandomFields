function [ W ] = getW( x, k )

val = 1/k;
N = size(x,1);
W = zeros(N,N);

[B,IDX] = sort(squareform(pdist(x)),2);

IDX = IDX(:,2:k+1);


inddd=sub2ind(size(W),repmat((1:length(x))',k,1), IDX(:));
W(inddd)=val;
sum(W,2);
end

