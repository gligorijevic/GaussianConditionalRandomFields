function  [similarity] = similarityNancy(JelenaData, k, n, t, N)
history = JelenaData(t-n+2:t+1, 1:N);
dist = exp(-k*pdist(history','euclidean').^2);
%corr = 1- pdist(history','correlation');
%sim = dist.*exp(corr-1);
%similarity = squareform(sim);
similarity = squareform(dist);
similarity(sum(history)<3, sum(history)<3) = 0;

similarity = similarity + eye(size(history,2));


%similarity = zeros(length(hosp_ids));

%for i=1:length(hosp_ids)
%    for j=i+1:length(hosp_ids)
%        similarity(i,j) = Node_Similarity(hosp_ids(i),hosp_ids(j),years(i),years(j),data_matrix);
%    end;
%end;
%similarity = similarity + similarity';

        