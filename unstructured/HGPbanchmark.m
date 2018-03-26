x_trening=x(:,:,trainigTS);
x_test=x(:,:,testTS);

y_trening=y(:,trainigTS);
y_testHGP=y(:,testTS);

xx_sve = reshape(ipermute(x,[1 3 2]),[],no_features);
sigma=300;

W =  exp(-squareform(pdist(xx_sve))/2*sigma);

yy_sve = y(:);%NxT

ylab=y_trening(~isnan(y_trening(:)));

[fu, fu_CMN] = harmonic_function(W,ylab);%fl=ylab;
predictions_HGP=fu(end-length(testTS)*N+1:end);

[ r2HGF,mseHGF, biasHFG ] = calculatePredictorPerformance( y_testHGP,predictions_HGP );


clearvars x_trening x_test y_trening y_testHGP xx_sve sigma yy_sve ylab nanValues