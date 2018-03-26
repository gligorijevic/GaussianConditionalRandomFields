function [ output_args ] = plotUPuncertainty( T, steps_ahead, y_plot, predictors, variances, N, i )

start_time = T(1);
end_time = T(2);

x_star = linspace(start_time, end_time + steps_ahead, end_time + steps_ahead);
x_star = x_star(:);
%% plot LR lag0 unvertainty
h = [0,0];
h(1) = subplot(2,3,1);

% predictionCRF(i:N:end);
Var = variances{1}(i,:);

fill([x_star; flipdim(x_star,1)], ...
    [(predictors{1}(i, :) + 1.96 * sqrt(Var))'; ...
    flipdim(predictors{1}(i,:) - 1.96*sqrt(Var), 2)'], ...
    [7 7 7] / 8 );
xlabel('Timesteps');
ylabel('Prediction with confidence interval');
hold on;

plot(start_time:end_time + steps_ahead, predictors{1}(i, :), ...
    'ro--','LineWidth',1.5);
plot(start_time:end_time + steps_ahead, y_plot,'x' ...
                              ,'MarkerFaceColor','g'...     
                              ,'LineWidth'      , 2 ...     
                              ,'MarkerSize'     ,10);
                          
hleg1 = legend('uncertainty', 'mean', 'training values');
set(hleg1,'Location','NorthWest');
set(hleg1,'Interpreter','none');
title('Uncertainty LR lag0');

%% plot LR lag1 unvertainty
h = [0,0];
h(1) = subplot(2,3,2);

% predictionCRF(i:N:end);
Var = variances{2}(i,:);

fill([x_star; flipdim(x_star,1)], ...
    [(predictors{2}(i, :) + 1.96 * sqrt(Var))'; ...
    flipdim(predictors{2}(i,:) - 1.96*sqrt(Var), 2)'], ...
    [7 7 7] / 8 );
xlabel('Timesteps');
ylabel('Prediction with confidence interval');
hold on;

plot(start_time:end_time + steps_ahead, predictors{2}(i, :), ...
    'ro--','LineWidth',1.5);
plot(start_time:end_time + steps_ahead, y_plot,'x' ...
                              ,'MarkerFaceColor','g'...     
                              ,'LineWidth'      , 2 ...     
                              ,'MarkerSize'     ,10);
                          
hleg1 = legend('uncertainty', 'mean', 'training values');
set(hleg1,'Location','NorthWest');
set(hleg1,'Interpreter','none');
title('Uncertainty LR lag1');

%% plot LR lag2 unvertainty
h = [0,0];
h(1) = subplot(2,3,3);

% predictionCRF(i:N:end);
Var = variances{3}(i,:);

fill([x_star; flipdim(x_star,1)], ...
    [(predictors{3}(i, :) + 1.96 * sqrt(Var))'; ...
    flipdim(predictors{3}(i,:) - 1.96*sqrt(Var), 2)'], ...
    [7 7 7] / 8 );
xlabel('Timesteps');
ylabel('Prediction with confidence interval');
hold on;

plot(start_time:end_time + steps_ahead, predictors{3}(i, :), ...
    'ro--','LineWidth',1.5);
plot(start_time:end_time + steps_ahead, y_plot,'x' ...
                              ,'MarkerFaceColor','g'...     
                              ,'LineWidth'      , 2 ...     
                              ,'MarkerSize'     ,10);
                          
hleg1 = legend('uncertainty', 'mean', 'training values');
set(hleg1,'Location','NorthWest');
set(hleg1,'Interpreter','none');
title('Uncertainty LR lag2');

%% plot LR lag0 unvertainty
h = [0,0];
h(1) = subplot(2,3,4);

% predictionCRF(i:N:end);
Var = variances{4}(i,:);

fill([x_star; flipdim(x_star,1)], ...
    [(predictors{4}(i, :) + 1.96 * sqrt(Var))'; ...
    flipdim(predictors{4}(i,:) - 1.96*sqrt(Var), 2)'], ...
    [7 7 7] / 8 );
xlabel('Timesteps');
ylabel('Prediction with confidence interval');
hold on;

plot(start_time:end_time + steps_ahead, predictors{4}(i, :), ...
    'ro--','LineWidth',1.5);
plot(start_time:end_time + steps_ahead, y_plot,'x' ...
                              ,'MarkerFaceColor','g'...     
                              ,'LineWidth'      , 2 ...     
                              ,'MarkerSize'     ,10);
                          
hleg1 = legend('uncertainty', 'mean', 'training values');
set(hleg1,'Location','NorthWest');
set(hleg1,'Interpreter','none');
title('Uncertainty GP lag0');

%% plot LR lag0 unvertainty
h = [0,0];
h(1) = subplot(2,3,5);

% predictionCRF(i:N:end);
Var = variances{5}(i,:);

fill([x_star; flipdim(x_star,1)], ...
    [(predictors{5}(i, :) + 1.96 * sqrt(Var))'; ...
    flipdim(predictors{5}(i,:) - 1.96*sqrt(Var), 2)'], ...
    [7 7 7] / 8 );
xlabel('Timesteps');
ylabel('Prediction with confidence interval');
hold on;

plot(start_time:end_time + steps_ahead, predictors{5}(i, :), ...
    'ro--','LineWidth',1.5);
plot(start_time:end_time + steps_ahead, y_plot,'x' ...
                              ,'MarkerFaceColor','g'...     
                              ,'LineWidth'      , 2 ...     
                              ,'MarkerSize'     ,10);
                          
hleg1 = legend('uncertainty', 'mean', 'training values');
set(hleg1,'Location','NorthWest');
set(hleg1,'Interpreter','none');
title('Uncertainty GP lag1');

%% plot LR lag0 unvertainty
h = [0,0];
h(1) = subplot(2,3,6);

% predictionCRF(i:N:end);
Var = variances{6}(i,:);

fill([x_star; flipdim(x_star,1)], ...
    [(predictors{6}(i, :) + 1.96 * sqrt(Var))'; ...
    flipdim(predictors{6}(i,:) - 1.96*sqrt(Var), 2)'], ...
    [7 7 7] / 8 );
xlabel('Timesteps');
ylabel('Prediction with confidence interval');
hold on;

plot(start_time:end_time + steps_ahead, predictors{6}(i, :), ...
    'ro--','LineWidth',1.5);
plot(start_time:end_time + steps_ahead, y_plot,'x' ...
                              ,'MarkerFaceColor','g'...     
                              ,'LineWidth'      , 2 ...     
                              ,'MarkerSize'     ,10);
                          
hleg1 = legend('uncertainty', 'mean', 'training values');
set(hleg1,'Location','NorthWest');
set(hleg1,'Interpreter','none');
title('Uncertainty GP lag2');


%% plot gcrf_new uncertainty
% h(2) = subplot(2,1,2);
% Var_new = Sigma_new(i:N:end);
% 
% fill([x_star; flipdim(x_star,1)], ...
%     [(predictionCRF_new(i:N:end) + 1.96 * sqrt(Var_new)); ...
%     flipdim(predictionCRF_new(i:N:end) - 1.96 * sqrt(Var_new), 1)], ...
%     [7 7 7] / 8 );
% xlabel('Timesteps')
% ylabel('Uncertainty')
% hold on;
% 
% plot(start_time:end_time + steps_ahead, predictionCRF_new(i:N:end), ...
%     'ro--','LineWidth',1.5);
% plot(start_time:end_time + steps_ahead, y_plot(i:N:end),'x'...
%                               , 'MarkerFaceColor','g'...          
%                               ,'LineWidth'      , 2 ...     
%                               ,'MarkerSize'     ,10);
% 
% hleg2 = legend('uncertainty','mean','training values');
% set(hleg2,'Location','NorthWest');
% set(hleg2,'Interpreter','none');
% title('Uncertainty estimation of new model');



end

