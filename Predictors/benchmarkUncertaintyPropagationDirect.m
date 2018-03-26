function [ predictors, variances] = ...
    benchmarkUncertaintyPropagationDirect( ...
    xtrain, lagtrain, utrain, ytrain, xvalid, lagvalid, uvalid, yvalid,...
    N, lag, trainTs, predictTs, select_features, training_window)

R_linreg = NaN(N, trainTs+predictTs);
Var_linreg = NaN(N, trainTs+predictTs);

%% Start for loop for each node in the graph
for i = 1:N
    %% Prepare training and testing inputs
    xx_train = squeeze(xtrain(i,1:trainTs,:));
    yy_train = ytrain(i,1:trainTs);
    
    %%  Train Linear ARX
    linear_mdl = LinearModel.fit(xx_train, yy_train);
    % Predict using Linear ARX
    R_linreg(i,1:trainTs) = predict(linear_mdl, xx_train);
    
    for lrTs = 1:trainTs
        Var_linreg(i,lrTs) = ...
            sum((yy_train(lrTs) - predict(linear_mdl, xx_train(lrTs,:))).^2)/...
            abs(size(xx_train(lrTs,:), 1) - size(xx_train(lrTs,:), 2));
    end
    
    %% Direct multiple-step-ahead prediction for Linear Regression
    xx_test = squeeze(xvalid(i,:,:));
    
    R_linreg(i, trainTs+1:end) = predict(linear_mdl, xx_test);
    Var_linreg(i, trainTs+1:end) = ...
        Var_linreg(i, trainTs) .* ...
        diag(1 + xx_test/(xx_train' * xx_train) * xx_test');
    
end

predictors{1} = R_linreg;
variances{1} = Var_linreg;

end

