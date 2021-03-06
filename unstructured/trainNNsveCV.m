x_trening=x(:,:,trainingTS);
x_test=x(:,:,testTS);
y_trening=y(:,trainingTS);
y_test=y(:,testTS);

xx = reshape(ipermute(x,[1 3 2]),[],no_features);
yy = y(:);%NxT


permutation = randperm(size(xx,1));
xxrand = xx(permutation,:);
yyrand = yy(permutation);
fold=5;
R2 = zeros(fold,1);
Rmodel=zeros(size(yy));
Rmodelrand=zeros(size(yy));
for iterval=1:fold
    iterval
    tresh_down=round((iterval-1)*(1/fold)*totalTS*N+1)
    tresh_up=round(iterval*(1/fold)*totalTS*N)

    testx=xxrand(tresh_down:tresh_up,:);   % testsize=size(testx)
    ytest=yyrand(tresh_down:tresh_up);
    trening= [xxrand(1:(tresh_down-1),:); xxrand(tresh_up+1:end,:)];  %  trsize=size(trening)
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
Rmodel=reshape(Rmodel,len,[]);