function [ node_results ] = node_up_accuracy( predictors, y, N, T, steps_to_predict )
%% Calculating accuracy on a station level
yy =y(:);

mse_results = zeros(length(predictors),N);
r2_results = zeros(length(predictors),N);
mse_results_train = zeros(length(predictors),N);
r2_results_train = zeros(length(predictors),N);

for p = 1:length(predictors)
    
    prediction =  predictors{p};
    
    %Test accuracy
    ytrueTest = yy(T(2)*N+1:(T(2)+steps_to_predict)*N,1);
    nanValues=isnan(ytrueTest);
    ytrueTest(nanValues)=[];
    
    prediction_test = prediction(:, T(2)+1:T(2)+steps_to_predict);
    prediction_test = prediction_test(:);
    
    pred_count = size(prediction_test, 1);
    
    for i = 1:N
        
        mse_results(p,i) = mse(ytrueTest(i:N:pred_count,1) - prediction_test(i:N:pred_count,1));
        
        r2_results(p,i) = 1 - (sum((ytrueTest(i:N:pred_count,1) - prediction_test(i:N:pred_count,1)).^2))/sum((ytrueTest(i:N:pred_count,1) - repmat(mean(ytrueTest(i:N:pred_count,1)), steps_to_predict, 1)).^2);
        
    end
    
    %Train accuracy
    ytrueTr = yy((T(1)-1)*N+1:T(2)*N,1);
    nanValues=isnan(ytrueTr);
    ytrueTr(nanValues)=[];
    
    prediction_train = prediction(:, T(1):T(2));
    prediction_train = prediction_train(:);
    
    pred_count = size(prediction_train, 1);
    
    for i = 1:N
        
        mse_results_train(p,i) = mse(ytrueTr(i:N:pred_count,1) - prediction_train(i:N:pred_count,1));
        
        r2_results_train(p,i) = 1 - (sum((ytrueTr(i:N:pred_count,1) - prediction_train(i:N:pred_count,1)).^2))/sum((ytrueTr(i:N:pred_count,1) - repmat(mean(ytrueTr(i:N:pred_count,1)), T(2)-T(1)+1,1)).^2);
        
    end
    
end

node_results = struct();
node_results.mse_results = mse_results;
node_results.r2_results = r2_results;
node_results.mse_results_train = mse_results_train;
node_results.r2_results_train = r2_results_train;

end

