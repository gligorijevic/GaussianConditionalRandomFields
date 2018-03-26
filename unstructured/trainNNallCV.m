RPredictions = R(:,testTS);
R2_R = 1 - sum((RPredictions - ytrue).^2)/sum((ytrue - mean(ytrue)).^2)


%%   
x_trening=x(:,:,trainigTS);
x_test=x(:,:,testTS);
y_trening=y(:,trainigTS);
y_test=y(:,testTS);

xx = reshape(ipermute(x_trening,[1 3 2]),[],no_features);
yy = y_trening(:);%NxT

permutation = randperm(size(xx,1));
xxrand = xx(permutation,:);
yyrand = yy(permutation);
fold=5;
R2 = zeros(fold,1);
Rmodel=zeros(size(yy));
Rmodelrand=zeros(size(yy));
for iterval=1:fold
    iterval
     tresh_down=round((iterval-1)*(1/fold)*trainigTS(end)*N+1)
     tresh_up=round(iterval*(1/fold)*trainigTS(end)*N)

    testx=xxrand(tresh_down:tresh_up,:);   
    ytest=yyrand(tresh_down:tresh_up);
    trening= [xxrand(1:(tresh_down-1),:); xxrand(tresh_up+1:end,:)];  
    ytre=[yyrand(1:(tresh_down-1)); yyrand(tresh_up+1:end)];
    tr= [trening, ytre];
    test=testx;
    [trash, net]=neural_simple(tr,[],info);
    predictionsNEW = sim(net, test')'; 
    Rmodelrand(tresh_down:tresh_up)=predictionsNEW; %popunjavam ali random, vrati posle u ne random
    R2_R = 1 - sum((ytest- predictionsNEW).^2)/sum((ytest- mean(ytest)).^2);
    R2(iterval)=R2_R;
end
R2;
Rmodel(permutation)=Rmodelrand;
% Rmodel=Rmodelrand;
Rmodel=reshape(Rmodel,len,[]);


tr= [xx, yy];
[trash, net]=neural_simple(tr,[],info);
predictionsNEW = sim(net, x_test')';
Rmodel=[Rmodel predictionsNEW];