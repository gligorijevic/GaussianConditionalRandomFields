clear all
close all 
clc 

% months_tr = 1:1:480;
% months_test = 481:1:708;

months_tr = 1:4;%u kolonama
months_test = 5;
len=900
lenmiss=4*900;
percentOfMissingdata=0;%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%define percent of missing data
% load attributes
% curr_dir = pwd;
%cd('C:\Users\jelena\Desktop\najnovije\FinalCRF_rain\Arifical_attributes');
% cd('C:\Users\jelena\Desktop\Graph Project\CODE\najnovije\missing\alpha  45 beta 6');%alpha  45 beta 6
%cd('C:\Users\jelena\Desktop\Graph Project\CODE\najnovije\notmissing\data2000x16')
load moze
% % load R
% % load similarity
% % y=y';
% % R=R';
y=y(1:len,months_tr:months_test); %%%%% proveri ovo kad ima vise od jednog koraka
R=R(1:len,months_tr:months_test);
similarity=similarity(1:len,1:len);
% size(y)
% percentOfMissingdata=0;
% y(0.8*len:len,:)=NaN;

% % % % % % %Generating missing data
% % % % % % indexesOfmissingData=randperm(lenmiss)';
% % % % % % indexes20=indexesOfmissingData(1: percentOfMissingdata*lenmiss,:);
% % % % % % y=y(:);
% % % % % % y(indexes20,1)=NaN;
% % % % % % y = reshape(y,len,months_tr(end));   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%PROMENI OVO 
pred{1} =R;
%create crf precision matrices

Data = struct;
%spatialScale = 100;
maxiter = 15;
% Data = precmatCRF(pred,y(:,months_tr),y(:,months_test),locStations(station_index,:),Data, 1,1, spatialScale);
Data = precmatCRF(pred,y(:,months_tr),y(:,months_test),similarity,Data, 1,1);
%%Test 

stepa = 0.1;
stepb = 0.1;

va = 0.5:stepa:2;
vb = 0.5:stepb:2;

[alpha,beta] = meshgrid(va,vb);
f = zeros(size(alpha));
gx = zeros(size(alpha));
gy = zeros(size(alpha));
mu = zeros(size(alpha));
%contour(v,v,f), hold on, quiver(v,v,-gx,-gy), hold off
 counter=0;
for i=1:size(alpha,1),
    for j=1:size(alpha,2),
        counter=counter+1
        alphaV=log(alpha(i,j));
        betaV=log(beta(i,j));
        u=[alphaV betaV];
        [funk g muk Q]=objectiveCRF(u,Data);
        f(i,j) = -funk;
        gx(i,j) = -g(1:1)/alpha(i,j);
        gy(i,j) = -g(2:end)/beta(i,j);
    end
end

[agx,agy] = gradient(f,stepa,stepb);

erroralpha = (mean(mean(abs(gx-agx))))
errorbeta = mean(mean(abs(gy-agy)))
error = (erroralpha+errorbeta)/2