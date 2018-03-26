% neural_simple.m	09/15/99
% express using of neural network for regression


function [pred, net]=neural_simple(trn,tst,info)
s=size(trn);
% block=[trn;val];
hidd=info.hidd;
%net=newff([min(block(:,1:s(2)-1));max(block(:,1:s(2)-1))]',[hidd 1],{'logsig','purelin'},'trainlm');
net=feedforwardnet(hidd,'trainlm');	
%net=feedforwardnet(hidd,'trainrp');	
%net=feedforwardnet(hidd,'traingd');	
%net=feedforwardnet(hidd,'traingdm');	
%net=feedforwardnet(hidd,'trainbr');	

net.layers{1}.transferFcn = 'logsig';%purelin,logsig, tansig
net.layers{2}.transferFcn = 'purelin';
% net.layers{3}.transferFcn = 'purelin';
% net.layers{4}.transferFcn = 'purelin';
% net.layers{5}.transferFcn = 'purelin';

% VV.P=val(:,1:s(2)-1)';
% VV.T=val(:,s(2))';
if isempty(tst)==0
   TV.P=tst(:,1:s(2)-1)';
   TV.T=tst(:,s(2))';
end
net.trainParam.epochs = info.epochs;
net.trainParam.show = info.show;
net.trainParam.max_fail=info.max_fail;
net.trainParam.show = NaN;
net.trainParam.showWindow = false;

%[net, tr] = train(net,trn(:,1:s(2)-1)',trn(:,s(2))',[],[],VV,TV);
[net, tr] = train(net,trn(:,1:s(2)-1)',trn(:,s(2))');

if isempty(tst)==0
   pred = sim(net,tst(:,1:s(2)-1)')'*stdv(s(2))+meanv(s(2))*ones(length(tst(:,1)),1);
else
   pred=[];
end

return;
  