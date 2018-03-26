% Load data and similarities here

% % % % % load 'sid_data_target_sve';
% % % % % load 'similarity01';
% % % % % 
% % % % % index_delete = [];
% % % % % for i = 1: size(sid_data_target,1)
% % % % %     aux = isnan(sid_data_target(i,:));
% % % % %     aux2= sum(sid_data_target(i,:));
% % % % %     if (or(sum(aux) > 0,aux2==0))
% % % % %         index_delete = [index_delete; i];
% % % % %     end;
% % % % %     
% % % % % end
% % % % % 
% % % % % sid_data_target(index_delete,: )=[];
% % % % % similarity(index_delete,:,:)=[];
% % % % % similarity(:,index_delete,:)=[];
% % % % % 
% % % % % for i = 1:size(similarity,3)
% % % % %     for j = 1: size(similarity, 1)
% % % % %         Norm = similarity(:,:,i);
% % % % %         Norm(Norm<prctile(prctile(Norm, 90, 1), 90, 2)) = 0;
% % % % %     end
% % % % % end

Data = y';
% aa = sid_similarity;
% 
% for t = 1:size(aa,3)
% %     sid_similarity1(:,:,t) = squeeze(sid_similarity(1:253,1:253,t))./repmat(sum(squeeze(sid_similarity(1:253,1:253,t)),2), 1, 253);
%     aaa(:,:,t) = squeeze(aa(1:253,1:253,t))./(sum(sum(squeeze(aa(1:253,1:253,t))))*0.5);
% end
% 

% aaa(index_delete,:,:)=[];
% aaa(:,index_delete,:)=[];

TemporalSimilarityMatrix = permute(similarity, [3,1,2]);

% Our data was in the following format: TxN matrix, each row representing a
% month and each column a paper. The values in the matrix were citation
% counts for that paper at that month. 

% The similarities we precomputed were reordered into row form, and in the
% case of temporal similarities, the time steps were stacked by row. So at
% row 1 we had the similarities between each pair of the N papers in
% question at T = 1, at row 2 it was the similarities at T = 2, etc.

% Our full data was bigger than the segment we focused on, this extracts
% the portions of interest from the full data.
T1 = 1;
T2 = 24;
N = 253;

Y = Data(T1:T2,1:N);
varianceY = var(Y(1:end));

[timeSteps papers] = size(Y);

[xx,yy] = meshgrid(1:papers,1:papers); 

finalSimilarityMatrix = [];
finalVarpair = [];

% Number of bins the entire set of values gets binned into; we used a basic
% binner which seems to fail when the number of bins is too big, so you may
% need to adjust this parameter to get it to work.
numberOfBins = 10;

tic

% This loop basically goes through all the timesteps of interest and
% aggregates the variance values into one vector
for timestep = 1:timeSteps
    % TemporalSimilarityMatrix is the matrix for the similarity you want to
    % plot. If the similarity is nontemporal, get rid of the 'timestep'
    % parameter; if you want to plot the product of two or more
    % similarities, remember to use '.' for pairwise multiplication.
    % (example provided)
    localSimilarityMatrix = reshape(TemporalSimilarityMatrix(timestep, :), papers, papers);
    % Example of a product of temporal and nontemporal similarities.
    % localSimilarityMatrix = reshape(TemporalSimilarityMatrix(timestep, :) .* NonTemporalSimilarityMatrix(timestep, :), papers, papers);
    finalSimilarityMatrix = [finalSimilarityMatrix; localSimilarityMatrix(:)];

    varpair = (Y(timestep,xx) - Y(timestep,yy)).^2;
    varpair = reshape(varpair, papers, papers);
    finalVarpair = [finalVarpair; varpair(:)];
end

finalVarpair = finalVarpair(find(finalSimilarityMatrix));
finalSimilarityMatrix = finalSimilarityMatrix(find(finalSimilarityMatrix));
F = ceil(numberOfBins * tiedrank(finalSimilarityMatrix) / length(finalSimilarityMatrix));
d_bin = accumarray(F,finalSimilarityMatrix);
var_bin = accumarray(F,finalVarpair);
[uniques,numUnique] = count_unique(F,'int');
var_bin = var_bin./(2.*numUnique);
d_bin = d_bin./numUnique;
toc

hold

% format the plot
plot(d_bin,var_bin, 'color', 'black', 'LineWidth', 2);
plot(d_bin,varianceY*ones(size(d_bin)),'color', 'black', 'LineWidth', 2);
xlabel('Similarity Name', 'FontSize', 20, 'FontName', 'Times New Roman')
set(gca, 'FontSize', 20, 'FontName', 'Times New Roman')
set(gcf, 'color', 'w')
ylabel('Variance', 'FontSize', 20, 'FontName', 'Times New Roman')
