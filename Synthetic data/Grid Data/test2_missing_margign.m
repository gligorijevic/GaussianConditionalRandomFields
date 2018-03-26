
clear all 
close all
clc 



rng(10);  % set random seed
numTimeSteps =12;
T=[1 numTimeSteps-1];
testcol=numTimeSteps
N = 400;  %have to be square of some number for grid and web structures
no_features = 30;
no_hiddenNeurons=40;
alpha =400;%bilo 100
beta =400;
sparseness = 1;
minSimilarityValue=0.5;
maxSimilarityValue=1;
xmin=0;
xmax=1;%0.5;

%sparseness = 1-(log(N)/log(1.06))/N;
maxiter = 50;

[y similarity R x alpha beta theta] = synthesize_data(numTimeSteps,N,alpha, beta, sparseness,no_features, no_hiddenNeurons,minSimilarityValue,maxSimilarityValue, xmin, xmax);



%% crtanje sa gridom
i=1:N;
i=int32(i);
L=sqrt(N);
L=int32(L);
yax=idivide(i,L);
yax=yax+1;
xax=mod(i,L);
% proverastaro=[i' xax' yax'];
yax(xax==0)=yax(xax==0)-1;
xax(xax==0)=L;
figure;

subplot(2,2,1)
scatter(xax, yax, [], R((yax-1)*L+xax,testcol),'filled')
title('R')


subplot(2,2,2)
scatter(xax, yax, [], y((yax-1)*L+xax,testcol),'filled')
title('y')
yTrue = y(:, testcol);
RPredictions = R(:, testcol);

figure

scatter(1:N,yTrue,'g')
hold all
scatter(1:N,RPredictions,'r')
% hold all
% scatter(similar,similarity(triu(similarity)>0),'sy')
title('drugi R nau?eni kroz validacije-red CRF-blue Y-green')



% % % % % %% nn parameters
% % % % % info.hidd = 45;%3*no_hiddenNeurons/2;			% number of hidden neurons
% % % % % info.epochs = 15;	% max number of training iterations (epochs)
% % % % % info.show = NaN;		% show training results each 'show' epochs
% % % % % info.max_fail = 5;	% if error does not decrease on 'val' set in 
% % % % %                      % 'max_fail' consecutive epochs, stop the training
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %% learning curve parameters
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % count=1:N;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % count=count';
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % similar=1.5:1:N;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % korak=sqrt(N)*2
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % tester=2*korak:korak:N;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %% 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % yTrue = y(:, testcol);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % RPredictions = R(:, testcol);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % R2_R = 1 - sum((RPredictions - yTrue).^2)/sum((yTrue - mean(yTrue)).^2)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % xx = reshape(ipermute(x,[1 3 2]),[],no_features);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % xx = zscore(xx);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % yy = y(:);%NxT
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % permutation = randperm(size(xx,1));
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % xxrand = xx(permutation,:);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % yyrand = yy(permutation);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % aaa=unique(sort(  kron(tester, T(1):T(2))))
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % mseunstr = zeros(length(aaa),1);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % si=zeros(length(aaa),1);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % counter =0;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % for sample=aaa
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %    
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     coefs = regress(yy(1:sample),xx(1:sample,:));
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     predictions = xx(T(2)*N+1:end,:) * coefs;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         tr= [xx(1:sample,:), yy(1:sample)];
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     test=xx((T(2)*N+1):end,:);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     [trash, net]=neural_simple(tr,[],info);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     predictions = sim(net, test')';
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %   R2_R = 1 - sum((predictions - yy(T(2)*N+1:end)).^2)/sum((yy(T(2)*N+1:end) - mean(yy(T(2)*N+1:end))).^2);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %R2_R = 1 - sum((predictions - yy(1:sample)).^2)/sum((yy(1:sample) - mean(yy(1:sample))).^2);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     mseNN = mse(predictions-yTrue);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     counter = counter+1
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     si(counter)=sample;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     mseunstr(counter)=mseNN;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % plot(si,mseunstr)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % fold=5;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % R2 = zeros(fold,1);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % Rmodelrand=zeros(size(yy));
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % for iterval=1:fold
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     iterval
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     tresh_down=round((iterval-1)*(1/fold)*numTimeSteps*N+1)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     tresh_up=round(iterval*(1/fold)*numTimeSteps*N)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     testx=xxrand(tresh_down:tresh_up,:);   % testsize=size(testx)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     ytest=yyrand(tresh_down:tresh_up);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     trening= [xxrand(1:(tresh_down-1),:); xxrand(tresh_up+1:end,:)];  %  trsize=size(trening)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     ytre=[yyrand(1:(tresh_down-1)); yyrand(tresh_up+1:end)];
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     tr= [trening, ytre];
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     test=testx;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     [trash, net]=neural_simple(tr,[],info);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     predictionsNEW = sim(net, test')';
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     coefs = regress(ytre,trening);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     predictionsNEW = testx*coefs;   
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     Rmodelrand(tresh_down:tresh_up)=predictionsNEW; %popunjavam ali random, vrati posle u ne random
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     R2_R = 1 - sum((ytest- predictionsNEW).^2)/sum((ytest- mean(ytest)).^2);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     R2(iterval)=R2_R;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % R2;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % Rmodel(permutation)=Rmodelrand;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % Rmodel=Rmodelrand;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % Rmodel=reshape(Rmodel,N,[]);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % y=y';
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % Rmodel=Rmodel';
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % MseCirf = zeros(length(tester),1);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % MseNNcrf = zeros(length(tester),1);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % lcR2CRF=zeros(length(tester),1);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % lcR2NN=zeros(length(tester),1);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % sicrf=zeros(length(tester),1);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % counterCRF=0;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %% learning curve za CRF i NN
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % for sample=tester
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     nesto{1}=Rmodel(1:sample,T(1):testcol);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %nesto{1}=predictions(1:sample,:);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     Data = struct;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     Data = precmatCRF(nesto,y(1:sample,T(1):T(2)),y(1:sample,testcol),similarity(1:sample, 1:sample),Data, 1,1);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     [ualpha ubeta pred Q]= trainCRF(Data,maxiter);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     predictionCRF= nezavisniTestCRF(Data,ualpha,ubeta);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     predictionNN=Rmodel(1:sample,testcol);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %[ualpha ubeta]= trainCRF(y(T(1):T(2),1:sample),Rmodel(T(1):T(2),1:sample),lagP,maxiter,similarity(1:sample,1:sample));
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %[predictionNN predictionCRF yTrue alpha beta] = testCRF(y(test,1:sample),Rmodel(test,1:sample),lagP,ualpha,ubeta, similarity(1:sample,1:sample));
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     yTrueovde = y(1:sample, testcol);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     R2_CRF = 1 - (sum((yTrueovde - predictionCRF).^2))/sum((yTrueovde - mean(yTrueovde)).^2) ; 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     R2_NN = 1 - (sum((yTrueovde - predictionNN).^2))/sum((yTrueovde - mean(yTrueovde)).^2); 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     mseCRF = mse(yTrueovde - predictionCRF);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     mseNN = mse(yTrueovde - predictionNN);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     counterCRF = counterCRF+1
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     sicrf(counterCRF)=sample*(numTimeSteps-1);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     MseCirf(counterCRF)=mseCRF;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     MseNNcrf(counterCRF)=mseNN;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     lcR2CRF(counterCRF)=   R2_CRF;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     lcR2NN(counterCRF)=   R2_NN;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % figure;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % subplot(3,1,1)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % plot(si,mseunstr)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % title(' Learning curve unstructured-a koji nema veze sa ovim dole jer se ovaj dole u?i onako po foldovima')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % subplot(3,1,2)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % plot(sicrf,MseCirf,'b')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % hold all
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % plot(sicrf,MseNNcrf,'r')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % title('CRF blue, unstructured red')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % subplot(3,1,3)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % plot(sicrf,lcR2CRF,'b')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % hold all
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % plot(sicrf,lcR2NN,'r')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % title('CRF blue, unstructured red')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %% krajni poziv CRF-a
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     nesto{1}=Rmodel(1:N,T(1):testcol);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     Data = struct;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     Data = precmatCRF(nesto,y(1:N,T(1):T(2)),y(1:N,testcol),similarity,Data, 1,1);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %%Test 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     [ualpha ubeta pred Q]= trainCRF(Data,maxiter);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % predictionCRF= testCRF(Data,ualpha,ubeta);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     predictionCRF= nezavisniTestCRF(Data,ualpha,ubeta);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     predictionNN=Rmodel(1:N,testcol);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %% crtanje
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % figure;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % subplot(2,1,1)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % scatter(count,RPredictions,'r')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % hold all
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % scatter(count,predictionCRF(:,1),'b')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % hold all
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % scatter(count,yTrue,'g')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % hold all
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % scatter(similar,similarity(triu(similarity)>0),'sy')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % title('prvi R iz sinteti?kih podataka-red    Y-green')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % subplot(2,1,2)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % scatter(count,predictionCRF,'b')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % hold all
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % scatter(count,yTrue,'g')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % hold all
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % scatter(count,predictionNN,'r')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % hold all
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % scatter(similar,similarity(triu(similarity)>0),'sy')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % title('drugi R nau?eni kroz validacije-red CRF-blue Y-green')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %% crtanje sa gridom
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % i=1:N;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % i=int32(i);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % L=sqrt(N);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % L=int32(L);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % yax=idivide(i,L);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % yax=yax+1;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % xax=mod(i,L);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % proverastaro=[i' xax' yax'];
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % yax(xax==0)=yax(xax==0)-1;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % xax(xax==0)=L;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % figure;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % subplot(2,2,1)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % scatter(xax, yax, [], R((yax-1)*L+xax,testcol),'filled')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % title('R')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % subplot(2,2,2)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % scatter(xax, yax, [], y((yax-1)*L+xax,testcol),'filled')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % title('y')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % subplot(2,2,3)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % scatter(xax, yax, [], predictionCRF((yax-1)*L+xax),'filled')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %  % % % % % % title('CRF')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % subplot(2,2,4)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % scatter(xax, yax, [], predictionNN((yax-1)*L+xax),'filled')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % title('NN')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %% krajni rezultati
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % alpha_nauceno=exp(ualpha)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % beta_nauceno=exp(ubeta)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % alpha_pravo=alpha
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % beta_pravo=beta
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % mseCRF = mse(yTrue - predictionCRF)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % mseNN = mse(yTrue - predictionNN)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % R2_CRF = 1 - (sum((yTrue - predictionCRF).^2))/sum((yTrue - mean(yTrue)).^2)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % R2_NN = 1 - (sum((yTrue - predictionNN).^2))/sum((yTrue - mean(yTrue)).^2)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % R = R';
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % y = y';
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % Rmodel = Rmodel';
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % [lcR2CRF lcR2NN MseCirf MseNNcrf]
clearvars -except  N T alpha beta R S x sparseness theta y similarity Rmodel MseCirf MseNNcrf lcR2CRF lcR2NN R2_R no_features numTimeSteps testcol predictionCRF predictionNN
