function [similarity] = generateGrid(N, minSimilarityValue, maxSimilarityValue)
%N=9;
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

    coordinate=[i' xax' yax' (xax' -yax')];
    sortcor=sortrows(coordinate,4);
    sortindexes=sortcor(:,1) ;
% provera=[i' xax' yax']

% figure
% scatter(yax,xax);
% grid on;

%% postavlja ke?eve tamo gde su veze
s=zeros(N);
for poy=1:N
   for pox=1:N
       razlikaPOx=abs(xax(poy)-xax(pox));
       razlikaPOy=abs(yax(poy)-yax(pox));
       if and(and(razlikaPOx<=1 , razlikaPOy<=1),abs(razlikaPOx-razlikaPOy)==1)  %% ovo je ako ho?u baš mrežu
       %if or(and(and(razlikaPOx<=1 , razlikaPOy<=1 ),abs(razlikaPOx-razlikaPOy)==1),and(razlikaPOx==1 , razlikaPOy==1 )) % a ovo je ako je povezan i sa dijagonalom
        s(poy, pox)=1;
       end
   end  
end

% % % % % %% vrednosti su razli?ite od jedinice (postavlja neke vrednosti tamo gde su jedinice) od najveceg do najmanjeg odozdo nagore
[indx, indy]=find(s);
newvalues=random('uniform', minSimilarityValue, maxSimilarityValue,length(indx),1);
newvalues=sort(newvalues);
%indsortx=sort(indx, 'descend');
%indsorty=sort(indy,'ascend');
A=[indx, indy];
[Y,I]=sort(A(1,:));
B=A(:,I); %use the column indices from sort() to sort all columns of A.


%%indices = sub2ind(size(s), indsortx, indsorty);
sortedindices=[];
for zbudz=sortindexes'
    %zbudz
    sortedindices=[sortedindices; B(and(B(:,1)==zbudz, B(:,1)<B(:,2)),:)];
end
indices = sub2ind(size(s), sortedindices(:,1), sortedindices(:,2));
%R(sortindexes,:)=blaj;
s(indices)=newvalues(1:length(sortedindices));
s = triu(s,1)+triu(s,1)'; 

%% vrednosti su razli?ite od jedinice (postavlja neke vrednosti tamo gde su jedinice
% % % % % % % % % % % % % % % % % [indx, indy]=find(s);
% % % % % % % % % % % % % % % % % newvalues=random('uniform', minSimilarityValue, maxSimilarityValue,length(indx),1);
% % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % indices = sub2ind(size(s), indx, indy);
% % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % s(indices)=newvalues;
% % % % % % % % % % % % % % % % % s = triu(s,1)+triu(s,1)'; 
%%
similarity=s;

% [Cmax,IndC]=max(sum(s));% najvise nosi linkova
