function [ normx, normy ] = normalize01( x,y )
%NORMALIZE01 Summary of this function goes here
%   Detailed explanation goes here

n=size(y,1);
t=size(y,2);
atr=size(x,2);

normy=nan(n,t);
normx=nan(n,atr,t);

for i =1:n
    xi=x(i,:,:);
    xi=squeeze(xi)';
    yi=y(i,:);
    
    ccc=xi-repmat(min(xi),t,1);
    vvv=repmat((max(xi)-min(xi)),t,1);
    vvv(vvv(:,:)==0)=1;
    ddd=ccc./vvv;
    nxi=ddd;
  %  nxi=((xi-repmat(min(xi),t,1)) ./ repmat((max(xi)-min(xi)),t,1));
    normx(i,:,:)=nxi';
    normy(i,:)=(yi-min(yi(:))) ./ (max(yi(:))-min(yi(:)));
end


% % % 
% % % normy=y(:);
% % % normx=reshape(ipermute(x,[1 3 2]),[],atr);
% % % 
% % % dataset=[normx normy];
% % % scaledI = (dataset-min(dataset(:))) ./ (max(dataset(:)-min(dataset(:))));
% % % min(scaledI(:)) % the min is 0
% % % max(scaledI(:)) % the max 1
% % % 
% % % 
% % % normy = reshape(scaledI(:,end),n,t);
% % % normx=reshape(scaledI(:,1:end-1),n,t, atr);
% % % normx=reshape(ipermute(normx,[1 3 2]),n,atr,t);

end

