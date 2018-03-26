function [ Q1, Q2, b ] = calcPrecision_uf( theta_alpha, theta_beta, CRFData, str )

if (strcmp(str,'training')==1)
    t = CRFData.Ttr;
end;

if (strcmp(str,'test')==1)
    t = CRFData.Ttr + CRFData.Ttest;
end;

nt = CRFData.N * t;
Q1 = sparse(nt, nt);
b = zeros(nt, 1);

for  i = 1:CRFData.noAlphas
    ualpha{i} = zeros(CRFData.N,t);
    for j= 1:t
        ualpha{i}(:,j) = [ ones(CRFData.N,1), CRFData.x(:,CRFData.alpha_features,j)] ...
            * theta_alpha((i-1)*(length(CRFData.alpha_features)+ 1) + 1 :...
            (i-1)*(length(CRFData.alpha_features) + 1) + length(CRFData.alpha_features) + 1)';
    end
end

for i =1:CRFData.noBetas
    ubeta{i} = zeros(CRFData.N,CRFData.N*t);
    for time=1:t
        %         beta_matrix = nan(CRFData.N,CRFData.N);
        %         for j= 1:CRFData.N
        %             for k = 1:CRFData.N
        %                 one = (i-1)*(size(CRFData.x,2) + 1) + 1
        %                 two = (i-1)*(size(CRFData.x,2) + 1) + size(CRFData.x,2) + 1
        %                 aux = [ ones(CRFData.N,1), abs(CRFData.x(:,:,t) - CRFData.x(:,:,t))] * ...
        %                     theta_beta((i-1)*(size(CRFData.x,2) + 1) + 1:...
        %                     (i-1)*(size(CRFData.x,2) + 1) + size(CRFData.x,2) + 1)';
        %
        %                 beta_matrix(j,:) = aux;
        %                 beta_matrix(:,k) = aux;
        %             end
        %         end
        theta_t = theta_beta((i-1)*length(CRFData.beta_features) + 1:...
            (i-1)*length(CRFData.beta_features) + length(CRFData.beta_features));
        xx_t = CRFData.x(:,CRFData.beta_features,time)*theta_t';
        [xx yy] = meshgrid(1:CRFData.N,1:CRFData.N);
        temp_sum = (xx_t(xx,:)-xx_t(yy,:));
        beta_matrix = reshape(exp(-sum(temp_sum,2)),CRFData.N,CRFData.N);
        
        ubeta{i}(:,(time-1)*CRFData.N +1:(time-1)*CRFData.N + CRFData.N) = beta_matrix;
    end
end

for i = 1:CRFData.noAlphas
    bb =  2 * exp(ualpha{i}) .* CRFData.predictors{i}(:, 1:t);
    b = b + bb(:);
    
    alpha_norm = exp(ualpha{i});
    alpha_norm = alpha_norm(:);
    Q1 = Q1 + diag(alpha_norm);
end;

Q2 = sparse(nt,nt);

for i=1:CRFData.noBetas
    Q2 = Q2 - repmat(exp(ubeta{i}), t,1) .* CRFData.betaMatrix{i}(1:nt, 1:nt);
    
    aux = full(sum(Q2,2));
    Q2diag = spconvert([(1:length(aux))', (1:length(aux))', aux]);
    Q2 = Q2 - Q2diag;
end;

end
