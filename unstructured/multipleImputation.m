    x_trening=x(:,:,trainigTS);
    y_treningORG=y(:,trainigTS);

    xx_trening = reshape(ipermute(x_trening,[1 3 2]),[],no_features);
    y_treningORG = y_treningORG(:);%NxT

    ylab=y_treningORG(~isnan(y_treningORG(:)));
    xlab=xx_trening(~isnan(y_treningORG(:)),:);
    x_unlab=xx_trening(isnan(y_treningORG(:)),:);



% % % %     meanfunc = @meanZero; hyp.mean = [];
% % % %     disp('covfunc = {@covMaterniso, 3}; ell = 1/4; sf = 1; hyp.cov = log([ell; sf]);')
% % % %     D=no_features;
% % % %     covfunc = @covSEard; L = ones(D,1);L = rand(D,1); sf = 1; hyp.cov = log([L; sf]);%L = rand(D,1)
% % % %     disp('likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);')
% % % %     likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);
% % % % 
% % % %     hyp = minimize(hyp, @gp, -100, @infExact, meanfunc, covfunc, likfunc, xlab, ylab);
% % % %     disp('[m s2] = gp(hyp, @infExact, meanfunc, covfunc, likfunc, x, y, z);')
% % % %     [m s2 nesto sigmee] = gp(hyp, @infExact, meanfunc, covfunc, likfunc, xlab, ylab, x_unlab);
% % % %     mean(sigmee)
    modelLinReg = LinearModel.fit(xlab, ylab);
    s2lab = sum((ylab - predict(modelLinReg, xlab)).^2 )/ (no_features - 2);

    m = predict(modelLinReg, x_unlab);
    s2= s2lab .* diag(1 + (x_unlab*inv(xlab'* xlab) * x_unlab'));
        
    sveualpha=zeros(10,1);sveubeta=zeros(10,1);
    yoriginalno=y;
tic    
rng('shuffle');
for i = 1:10
    ysample=y_treningORG(:);
%     if percentOfMissingdata==0
%         sample=[];
%     else
        sample=normrnd(m,sqrt(s2));
%     end
    ysample(isnan(y_treningORG(:)))=sample;
    ysample=reshape(ysample,len,[]);
    y=[ysample y(:,testTS)];
    
    trainNNsveCV
    RR{1} =Rmodel;
    % DemoSemiSVR

    %% create crf precision matrices
    Data = struct;
    %Data = precmatCRF(RR,y(:,trainigTS),y(:,testTS),similarity,Data, 1,1); 
    Data = precmatCRF_rain(RR,y(:,trainingTS),y(:,testTS),locStations(station_index,:),Data, 1,1, spatialScale);

    DataIgnore=Data;
    [ualphaIgnore ubetaIgnore predIgnore QIgnore]= trainCRFIgnore(DataIgnore,maxiter);
    sveualpha(i)=ualphaIgnore;
    sveubeta(i)=ubetaIgnore;
    

end

predictionCRFIgnore= nezavisniTestCRF(DataIgnore,mean(sveualpha),mean(sveubeta));
y=yoriginalno;
[r2_CRF_ignore,mse_CRF_ignore, biasTRUECRFIGNORE ] = calculatePredictorPerformance( y(:,testTS),predictionCRFIgnore )
alpha_ignore=mean(sveualpha);
beta_ignroe=mean(sveubeta);
ignorevreme=toc


