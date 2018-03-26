
y_testHGP=y(:,testTS);

xx_sve = reshape(ipermute(x,[1 3 2]),[],no_features);
yy_sve = y(:);%NxT
yy_sve(end-N+1:end)=NaN;
uu_index = isnan(yy_sve(:));
ss_index = ~isnan(yy_sve(:));


k = 5;
aa = 2;
bb = 3;

W =getW(xx_sve,k); %having 1/k if xj is among k nearest neighbors of xi, otherwise 0

options = optimset('Display','Iter','MaxIter',50,'GradObj','on'); %optimization 
options = optimset(options,'UseParallel','always');
x0 = [bb, aa];
uu = fminunc(@(u)objectiveSSGF(u, xx_sve, yy_sve, k),x0,options);  
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
predictionSSGF=yu(end-N+1:end);
MSE_HGF
[ r2HGF,mseHGF, biasHFG ] = calculatePredictorPerformance( y_testHGP,predictionSSGF )

% % % yy_sve(uu_index)=yu;
% % % 
% % % y_treningORG=reshape(y_treningORG, N, length(trainigTS));
% % % 
% % % y=[y_treningORG , y(:,testTS)]