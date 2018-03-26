function [similarities values] = temporalGraph(T,N,commSize,historyLength,meanDiff,sigma, k)
    similarities = zeros(N,N,T);
    values = zeros(N,T);
    alphas = random('uniform', 0, 1, N, historyLength);
    ss = repmat(sum(alphas, 2), 1, size(alphas,2));
    alphas = alphas./ss;
    for i=1:historyLength
        [values(:,i) similarities(:,:,i)] = timeSlice(N, commSize, meanDiff, sigma, k);
    end
    
    for i=historyLength+1:T
%         [v sim] = timeSlice(N, commSize, meanDiff, sigma, k);
%         similarities(:,:,i) = sim;
%         ss = repmat(sum(sim, 2), 1, size(sim,2));
%         sim = sim./ss;
%         values(:,i) = sim*sum(alphas .* values(:,i-historyLength:i-1), 2);        
        values(:,i) = sum(alphas .* values(:,i-historyLength:i-1), 2) + random('uniform', -5*sigma, 5*sigma, N, 1);
         similarities(:,:,i) =  exp(-k*squareform(pdist(values(:,i),'euclidean').^2));
    end
end
