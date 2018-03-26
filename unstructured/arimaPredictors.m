function [ predictors, variances, confidence_guesses, arima_params ] = arimaPredictors( y, yGhost, numTimeSteps, N, T, training_window )
%ARIMAPREDICTORS Summary of this function goes here
%   Detailed explanation goes here
% Condition is that one interval in CV has to be 12 months long!!!!!

no_diseases = size(y,1);
foo = size(y,2);

R_arima1 = zeros(no_diseases, foo);
R_arima2 = zeros(no_diseases, foo);
R_arima3 = zeros(no_diseases, foo);
Var_arima1 = zeros(no_diseases, foo);
Var_arima2 = zeros(no_diseases, foo);
Var_arima3 = zeros(no_diseases, foo);

arima_params = zeros(no_diseases, 11);

for i = 1:no_diseases
    
    fold = 5;
    
    yi=y(i,T(1):T(2))';
    
    for iterval=1:fold
        
        iterval;
        tresh_down = round((iterval-1) * (1/fold) * T(2) + 1);
        tresh_up = round(iterval * (1/fold) * T(2));
        
        disp(['Node: ', num2str(i),' is on ',num2str(iterval),' step of crossvalidation, data is taking steps from: ',num2str(tresh_down),' to: ',num2str(tresh_up)]);
        
        %Training X and Y
        yy_cv_test = yi(tresh_down:tresh_up);
        
        if tresh_down == 1
            yy_cv_test = [yGhost(i,end-1:end)'; yy_cv_test];
        else
            yy_cv_test = [yi(tresh_down-2:tresh_down-1); yy_cv_test];
        end
        
        ytre=[yGhost(i,:)'; yi(1:(tresh_down-1))];
        ytre = ytre(end-training_window+1:end);
        
        Y = ytre;
        N = length(Y);
        if sum(Y) ~= 0
            model1 = arima('Constant',0,'D',1,'Seasonality',3,'MALags',1,'SMALags',3);
            model2 = arima('Constant',0,'D',1,'Seasonality',6,'MALags',1,'SMALags',6);
            model3 = arima('Constant',0,'D',1,'Seasonality',12,'MALags',1,'SMALags',12);
            fit1 = estimate(model1,Y);
            fit2 = estimate(model2,Y);
            fit3 = estimate(model3,Y);
            
%             for t = 1:12
%                 
%                 [R_arima1(i,tresh_down+t-1), Var_arima1(i,tresh_down+t-1)] = forecast(fit1,1,'Y0',Y);
%                 [R_arima2(i,tresh_down+t-1), Var_arima2(i,tresh_down+t-1)] = forecast(fit2,1,'Y0',Y);
%                 [R_arima3(i,tresh_down+t-1), Var_arima3(i,tresh_down+t-1)] = forecast(fit3,1,'Y0',Y);
%             end
            
            
                        [Yf1,YMSE1] = forecast(fit1,12,'Y0',Y);
                        [Yf2,YMSE2] = forecast(fit2,12,'Y0',Y);
                        [Yf3,YMSE3] = forecast(fit3,12,'Y0',Y);
                        R_arima1(i,tresh_down:tresh_up) = Yf1;
                        R_arima2(i,tresh_down:tresh_up) = Yf2;
                        R_arima3(i,tresh_down:tresh_up) = Yf3;
                        Var_arima1(i,tresh_down:tresh_up) = YMSE1;
                        Var_arima2(i,tresh_down:tresh_up) = YMSE2;
                        Var_arima3(i,tresh_down:tresh_up) = YMSE3;
        end
    end
    
    Y = y(i,T(2)-training_window+1:T(2))';
    N = length(Y);
    if sum(Y) ~= 0
        model1 = arima('Constant',0,'D',1,'Seasonality',3,'MALags',1,'SMALags',3);
        model2 = arima('Constant',0,'D',1,'Seasonality',6,'MALags',1,'SMALags',6);
        model3 = arima('Constant',0,'D',1,'Seasonality',12,'MALags',1,'SMALags',12);
        fit1 = estimate(model1,Y);
        fit2 = estimate(model2,Y);
        fit3 = estimate(model3,Y);
        
        arima_params(i,1) = fit1.P;
        arima_params(i,2) = fit1.Q;
        arima_params(i,3) = fit1.D;
        arima_params(i,5) = fit1.P;
        arima_params(i,6) = fit1.Q;
        arima_params(i,7) = fit1.D;
        arima_params(i,9) = fit1.P;
        arima_params(i,10) = fit1.Q;
        arima_params(i,11) = fit1.D;
        
        
                [Yf1,YMSE1] = forecast(fit1,12,'Y0',Y);
                [Yf2,YMSE2] = forecast(fit2,12,'Y0',Y);
                [Yf3,YMSE3] = forecast(fit3,12,'Y0',Y);
                R_arima1(i,tresh_up+1:end) = Yf1;
                R_arima2(i,tresh_up+1:end) = Yf2;
                R_arima3(i,tresh_up+1:end) = Yf3;
                Var_arima1(i,tresh_up+1:end) = YMSE1;
                Var_arima2(i,tresh_up+1:end) = YMSE2;
                Var_arima3(i,tresh_up+1:end) = YMSE3;
        
        %         [Yf,YMSE] = forecast(fit,12,'Y0',Y);
        %         R_arima(i,tresh_up+1:end) = Yf;
        %         Var_arima(i,tresh_up+1:end) = YMSE;
        
        
%         for t = 1:12
%             [R_arima1(i,tresh_down+t), Var_arima1(i,tresh_down+t-1)] = forecast(fit1,1,'Y0',Y);
%             [R_arima2(i,tresh_down+t), Var_arima2(i,tresh_down+t-1)] = forecast(fit2,1,'Y0',Y);
%             [R_arima3(i,tresh_down+t), Var_arima3(i,tresh_down+t-1)] = forecast(fit3,1,'Y0',Y);
%         end
        
    end
end



predictors{1} = R_arima1;
predictors{2} = R_arima2;
predictors{3} = R_arima3;

variances{1} = Var_arima1;
variances{2} = Var_arima2;
variances{3} = Var_arima3;

%% Calculating confidence interval guesses
confidence_interval_guesses = zeros(1,length(predictors));

yint = y(:, T(1): T(2));
yint = yint(:);

for i = 1 : size(yint,1)
    for j = 1 : length(predictors)
        pred = predictors{j}(:, T(1): T(2));
        pred = pred(:);
        sigma = variances{j}(:, T(1): T(2));
        sigma = sigma(:);
        if and(yint(i) > (pred(i) - 1.96*sqrt(sigma(i))), yint(i) < (pred(i) + 1.96*sqrt(sigma(i))))
            confidence_interval_guesses(1,j) = confidence_interval_guesses(1,j) + 1;
        end
    end
    
end

confidence_interval_guesses(confidence_interval_guesses==0)=confidence_interval_guesses(confidence_interval_guesses==0)+1; %ako bas nema ni jedan da kazemo da je bar jedan upao

confidence_guesses = confidence_interval_guesses./(N*(T(2)-T(1)+1));
printmat(confidence_guesses, 'Confidence interval guesses', 'percentage', 'arx_lag0 arx_lag1 arx_lag2 lasso_lag0 lasso_lag1 lasso_lag2 elnet_lag0 elnet_lag1 elnet_lag2 gp_lag0 gp_lag1 gp_lag2' )


end

