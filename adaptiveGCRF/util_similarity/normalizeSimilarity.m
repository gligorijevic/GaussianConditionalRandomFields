%% normalizovano po linkovima
for t=1:size(similarity,3)

    
    b=diag(similarity(:,:,t));
    c=repmat(b,1, size(similarity(:,:,t),1));
    d=repmat(b', size(similarity(:,:,t),1),1);
    
    similarity(:,:,t)=(similarity(:,:,t)./(c+d-similarity(:,:,t)));

end


% % 
% % % normalizovano na mesecnom nivou
% % for t=1:size(similarity,3)
% %     U = triu(similarity(:,:,t),1);
% %     nt=sum(sum(sum(U)));    
% %     similarity(:,:,t)=similarity(:,:,t)/nt;
% % end
