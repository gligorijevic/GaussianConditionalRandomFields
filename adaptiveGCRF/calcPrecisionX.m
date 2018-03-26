function [Q1, Q2, b] = calcPrecisionX(ualpha, ubeta, thetas_Rk, thetas_Sl, Data, str)


if (strcmp(str,'training')==1)
    t = Data.Ttr;
end;

if (strcmp(str,'test')==1)
    t = Data.Ttr + Data.Ttest;
end;

nt = Data.N*t;

Q1 = sparse(nt,nt);
b = zeros(nt,1);

% reordering input attributes for training
xx = Data.x(:, Data.xunstr, nts);
X = reshape(ipermute(xx,[1 3 2]), [], length(Data.xunstr));
X = [ones(size(X,1),1), X];
% y = Data.y(:, 1:Data.Ttr);

% x_train = squeeze(Data.x(:,:,1:t));
% for nts = 1:t
%     x_flatten = [x_flatten; squeeze(x_train(:,:,nts))];
% end
% x = [ones(nt, size(Data.x,2)), x_flatten];

for i=1:Data.nalpha 
    % bb =  2*exp(ualpha(i))*Data.R{i}(:,1:t);
    bb =  2 * exp(ualpha(i)) * X'*thetas_Rk;
    b = b + bb(:);
    Q1 = Q1 + exp(ualpha(i)) * Data.alphaMatrix{i}(1:nt,1:nt);
end

Q2 = sparse(nt,nt);

for i=1:Data.nbeta
    M0 = sparse(diag(ones(1,Data.T)));
    betaMatrix = kron(M0, Data.similarity{i});

    Q2 = Q2 - exp(ubeta(i)) * betaMatrix(1:nt,1:nt);
end

aux = full(sum(Q2,2));
Q2diag = spconvert([(1:length(aux))', (1:length(aux))', aux]);
Q2 = Q2 - Q2diag; 