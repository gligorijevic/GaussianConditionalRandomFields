% % % % % x_trening=x(:,:,trainingTS);
% % % % % x_test=x(:,:,testTS);
% % % % % y_trening=y(:,trainingTS);
% % % % % y_test=y(:,testTS);
% % % % % 
% % % % % 
% % % % % for i=1:size(x_trening,1)   
% % % % %     i
% % % % %     lenxtr=length(x_trening)  
% % % % %     yi = y(i,trainingTS)';
% % % % %     nonnan_values = ~isnan(yi);
% % % % %     xi = x_trening(i,:,:);
% % % % % 	xi_test=x_test(i,:,:);
% % % % % 	
% % % % % 	xxi = reshape(ipermute(xi,[1 3 2]),[],no_features);
% % % % % 	xxi_test=reshape(ipermute(xi_test,[1 3 2]),[],no_features);
% % % % % 	permutation = randperm(size(xxi,1));
% % % % % 	xxrand = xxi(permutation,:);
% % % % % 	yyrand = yi(permutation);
% % % % % 	fold=4;
% % % % % 	Rmodel=zeros(size(yi));
% % % % % 	Rmodelrand=zeros(size(yi));
% % % % % 	for iterval=1:fold
% % % % % 		iterval
% % % % % 		tresh_down=round((iterval-1)*(1/fold)*trainingTS(end)+1)
% % % % % 		tresh_up=round(iterval*(1/fold)*trainingTS(end))
% % % % % 
% % % % % 	% % %      tresh_down=round((iterval-1)*(1/fold)*trainingTS(end)*N+1)
% % % % % 	% % %      tresh_up=round(iterval*(1/fold)*trainingTS(end)*N)
% % % % % 
% % % % % 		testx=xxrand(tresh_down:tresh_up,:);   % testsize=size(testx)
% % % % % 		ytest=yyrand(tresh_down:tresh_up);
% % % % % 		trening= [xxrand(1:(tresh_down-1),:); xxrand(tresh_up+1:end,:)];  %  trsize=size(trening)
% % % % % 		ytre=[yyrand(1:(tresh_down-1)); yyrand(tresh_up+1:end)];
% % % % % 		tr= [trening, ytre];
% % % % % 		
% % % % % 		test=testx;
% % % % % 		[trash, net]=neural_simple(tr,[],info);
% % % % % 		predictionsNEW = sim(net, test')';
% % % % % 	%     coefs = regress(ytre,trening);
% % % % % 	%     predictionsNEW = testx*coefs;   
% % % % % 		Rmodelrand(tresh_down:tresh_up)=predictionsNEW; %popunjavam ali random, vrati posle u ne random
% % % % % 	end
% % % % % 	Rmodel(permutation)=Rmodelrand;
% % % % % 	Rmodel=reshape(Rmodel,1,[]);
% % % % % 	tr= [xxi, yi];
% % % % % 	[trash, net]=neural_simple(tr,[],info);
% % % % % 	predictionsNEW = sim(net, xxi_test')';
% % % % % 	Rmodel=[Rmodel predictionsNEW'];
% % % % % 	
% % % % %     uns(i,:) = Rmodel';
% % % % % end;
% % % % % % % % xx = reshape(ipermute(x_trening,[1 3 2]),[],no_features); %%XX=reshape(ipermute(S,[2 1 3]),[],8);
% % % % % % % % yy = y_trening(:);%NxT
% % % % % 
% % % % % RR{1} =uns;

total=length(trainingTS)+length(testTS);
pred = nan(length(y),total);
for i=1:length(y)   
    yi = y(i,trainingTS)';
 	xi = reshape(ipermute(x(i,:,trainingTS),[1 3 2]),[],no_features);
% % %     %xi =  squeeze(x(i,:,trainingTS));
% % %     %tst=squeeze(x(i,:,1:total));
     tr= [xi, yi];
     tst = reshape(ipermute(x(i,:,1:total),[1 3 2]),[],no_features);
% % %     
% % %     size(tst)
    [trash, net]=neural_simple(tr,[],info);
    pred(i,:) = sim(net, tst');
% % %     
    
    

% % % %      xi = [ones(length(trainingTS),1) reshape(ipermute(x(i,:,trainingTS),[1 3 2]),[],no_features)];
% % % %      w(:,i) = regress(yi,xi);
% % % %      x_pred =[ones(length(1:total),1) reshape(ipermute(x(i,:,1:total),[1 3 2]),[],no_features)];% [ones(size(X,1),1) squeeze(X(:,i,:))];
% % % %      pred(i,:) = x_pred*w(:,i);
    
% %     xi = [ones(length(trainingTS),1) squeeze(x(i,:,trainingTS))];
% %     w(:,i) = regress(yi,xi);
% %     x_pred =[ones(length(total),1) squeeze(x(i,:,1:total))];% [ones(size(X,1),1) squeeze(X(:,i,:))];
% %     pred(i,:) = x_pred*w(:,i);
end;

Rmodel=pred;
clearvars xi xxi yi w x_pred