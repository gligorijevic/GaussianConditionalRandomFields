%data
% % % % x = rand(10,3);
% % % % y = rand(10,1);
% % % % y([2,4,6]) = NaN;

    x_trening=x(:,:,trainigTS);
    y_treningORG=y(:,trainigTS);

    xx_trening = reshape(ipermute(x_trening,[1 3 2]),[],no_features);
    y_treningORG = y_treningORG(:);%NxT

    ylab=y_treningORG(~isnan(y_treningORG(:)));
    xlab=xx_trening(~isnan(y_treningORG(:)),:);
    x_unlab=xx_trening(isnan(y_treningORG(:)),:);




uu_index = isnan(y_treningORG(:));
ss_index = ~isnan(y_treningORG(:));

k = 4;
aa = 0.5;
bb = 1;

W =getW(xx_trening,k); %having 1/k if xj is among k nearest neighbors of xi, otherwise 0

options = optimset('Display','Iter','MaxIter',50,'GradObj','on'); %optimization 
options = optimset(options,'UseParallel','always');
x0 = [bb, aa];
uu = fminunc(@(u)objectiveSSGF(u, xx_trening, y_treningORG, k),x0,options);  
aa = uu(2);
bb = uu(1);

A = W + W' - W'*W;
I = eye(size(A));
L = I - A;
M = L + aa*I;

Muu = M(uu_index,uu_index);
Mus = M(uu_index,ss_index);
Mss = M(ss_index,ss_index);

yu = -inv(Muu)*Mus*y(ss_index);


y_treningORG(uu_index)=yu;

y_treningORG=reshape(y_treningORG, N, length(trainigTS));

y=[y_treningORG , y(:,testTS)]