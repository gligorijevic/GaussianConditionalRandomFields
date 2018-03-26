function [ ] = plotUPuncertainty2( T, steps_ahead, y_plot, predictors, variances, N, i )

start_time = T(1);
end_time = T(2)+steps_ahead;

start = 1;
finish = T(2) - T(1) + steps_ahead+1;

x_star = linspace(start, finish, finish);
x_star = x_star(:);
%% plot LR lag0 unvertainty
h = [0,0];
h(1) = subplot(4,3,1);

% predictionCRF(i:N:end);
Var = variances{1}(i,start_time:end_time);

fill([x_star; flipdim(x_star,1)], ...
    [(predictors{1}(i, start_time:end_time) + 1.96 * sqrt(Var))'; ...
    flipdim(predictors{1}(i,start_time:end_time) - 1.96*sqrt(Var), 2)'], ...
    [7 7 7] / 8 );
xlabel('Timesteps');
ylabel('Prediction with confidence interval');
hold on;

plot(start:finish, predictors{1}(i, start_time:end_time), ...
    'ro--','LineWidth',1.5);
plot(start:finish, y_plot,'x' ...
                              ,'MarkerFaceColor','g'...     
                              ,'LineWidth'      , 2 ...     
                              ,'MarkerSize'     ,10);
                          
hleg1 = legend('uncertainty', 'mean', 'training values');
set(hleg1,'Location','NorthWest');
set(hleg1,'Interpreter','none');
title('Uncertainty LR lag0');

%% plot LR lag1 unvertainty
h = [0,0];
h(1) = subplot(4,3,2);

% predictionCRF(i:N:end);
Var = variances{2}(i,start_time:end_time);

fill([x_star; flipdim(x_star,1)], ...
    [(predictors{2}(i, start_time:end_time) + 1.96 * sqrt(Var))'; ...
    flipdim(predictors{2}(i,start_time:end_time) - 1.96*sqrt(Var), 2)'], ...
    [7 7 7] / 8 );
xlabel('Timesteps');
ylabel('Prediction with confidence interval');
hold on;

plot(start:finish, predictors{2}(i, start_time:end_time), ...
    'ro--','LineWidth',1.5);
plot(start:finish, y_plot,'x' ...
                              ,'MarkerFaceColor','g'...     
                              ,'LineWidth'      , 2 ...     
                              ,'MarkerSize'     ,10);
                          
hleg1 = legend('uncertainty', 'mean', 'training values');
set(hleg1,'Location','NorthWest');
set(hleg1,'Interpreter','none');
title('Uncertainty LR lag1');

%% plot LR lag2 unvertainty
h = [0,0];
h(1) = subplot(4,3,3);

% predictionCRF(i:N:end);
Var = variances{3}(i,start_time:end_time);

fill([x_star; flipdim(x_star,1)], ...
    [(predictors{3}(i, start_time:end_time) + 1.96 * sqrt(Var))'; ...
    flipdim(predictors{3}(i,start_time:end_time) - 1.96*sqrt(Var), 2)'], ...
    [7 7 7] / 8 );
xlabel('Timesteps');
ylabel('Prediction with confidence interval');
hold on;

plot(start:finish, predictors{3}(i, start_time:end_time), ...
    'ro--','LineWidth',1.5);
plot(start:finish, y_plot,'x' ...
                              ,'MarkerFaceColor','g'...     
                              ,'LineWidth'      , 2 ...     
                              ,'MarkerSize'     ,10);
                          
hleg1 = legend('uncertainty', 'mean', 'training values');
set(hleg1,'Location','NorthWest');
set(hleg1,'Interpreter','none');
title('Uncertainty LR lag2');
%% plot Lasso lag0 unvertainty
h = [0,0];
h(1) = subplot(4,3,4);

% predictionCRF(i:N:end);
Var = variances{4}(i,start_time:end_time);

fill([x_star; flipdim(x_star,1)], ...
    [(predictors{4}(i, start_time:end_time) + 1.96 * sqrt(Var))'; ...
    flipdim(predictors{4}(i,start_time:end_time) - 1.96*sqrt(Var), 2)'], ...
    [7 7 7] / 8 );
xlabel('Timesteps');
ylabel('Prediction with confidence interval');
hold on;

plot(start:finish, predictors{4}(i, start_time:end_time), ...
    'ro--','LineWidth',1.5);
plot(start:finish, y_plot,'x' ...
                              ,'MarkerFaceColor','g'...     
                              ,'LineWidth'      , 2 ...     
                              ,'MarkerSize'     ,10);
                          
hleg1 = legend('uncertainty', 'mean', 'training values');
set(hleg1,'Location','NorthWest');
set(hleg1,'Interpreter','none');
title('Uncertainty Lasso lag0');
%% plot Lasso lag1 unvertainty
h = [0,0];
h(1) = subplot(4,3,5);

% predictionCRF(i:N:end);
Var = variances{5}(i,start_time:end_time);

fill([x_star; flipdim(x_star,1)], ...
    [(predictors{5}(i, start_time:end_time) + 1.96 * sqrt(Var))'; ...
    flipdim(predictors{5}(i,start_time:end_time) - 1.96*sqrt(Var), 2)'], ...
    [7 7 7] / 8 );
xlabel('Timesteps');
ylabel('Prediction with confidence interval');
hold on;

plot(start:finish, predictors{5}(i, start_time:end_time), ...
    'ro--','LineWidth',1.5);
plot(start:finish, y_plot,'x' ...
                              ,'MarkerFaceColor','g'...     
                              ,'LineWidth'      , 2 ...     
                              ,'MarkerSize'     ,10);
                          
hleg1 = legend('uncertainty', 'mean', 'training values');
set(hleg1,'Location','NorthWest');
set(hleg1,'Interpreter','none');
title('Uncertainty Lasso lag1');
%% plot Lasso lag2 unvertainty
h = [0,0];
h(1) = subplot(4,3,6);

% predictionCRF(i:N:end);
Var = variances{6}(i,start_time:end_time);

fill([x_star; flipdim(x_star,1)], ...
    [(predictors{6}(i, start_time:end_time) + 1.96 * sqrt(Var))'; ...
    flipdim(predictors{6}(i,start_time:end_time) - 1.96*sqrt(Var), 2)'], ...
    [7 7 7] / 8 );
xlabel('Timesteps');
ylabel('Prediction with confidence interval');
hold on;

plot(start:finish, predictors{6}(i, start_time:end_time), ...
    'ro--','LineWidth',1.5);
plot(start:finish, y_plot,'x' ...
                              ,'MarkerFaceColor','g'...     
                              ,'LineWidth'      , 2 ...     
                              ,'MarkerSize'     ,10);
                          
hleg1 = legend('uncertainty', 'mean', 'training values');
set(hleg1,'Location','NorthWest');
set(hleg1,'Interpreter','none');
title('Uncertainty Lasso lag2');
%% plot Elastic Net lag0 unvertainty
h = [0,0];
h(1) = subplot(4,3,7);

% predictionCRF(i:N:end);
Var = variances{7}(i,start_time:end_time);

fill([x_star; flipdim(x_star,1)], ...
    [(predictors{7}(i, start_time:end_time) + 1.96 * sqrt(Var))'; ...
    flipdim(predictors{7}(i,start_time:end_time) - 1.96*sqrt(Var), 2)'], ...
    [7 7 7] / 8 );
xlabel('Timesteps');
ylabel('Prediction with confidence interval');
hold on;

plot(start:finish, predictors{7}(i, start_time:end_time), ...
    'ro--','LineWidth',1.5);
plot(start:finish, y_plot,'x' ...
                              ,'MarkerFaceColor','g'...     
                              ,'LineWidth'      , 2 ...     
                              ,'MarkerSize'     ,10);
                          
hleg1 = legend('uncertainty', 'mean', 'training values');
set(hleg1,'Location','NorthWest');
set(hleg1,'Interpreter','none');
title('Uncertainty Elastic Net lag0');
%% plot Elastic Net lag1 unvertainty
h = [0,0];
h(1) = subplot(4,3,8);

% predictionCRF(i:N:end);
Var = variances{8}(i,start_time:end_time);

fill([x_star; flipdim(x_star,1)], ...
    [(predictors{8}(i, start_time:end_time) + 1.96 * sqrt(Var))'; ...
    flipdim(predictors{8}(i,start_time:end_time) - 1.96*sqrt(Var), 2)'], ...
    [7 7 7] / 8 );
xlabel('Timesteps');
ylabel('Prediction with confidence interval');
hold on;

plot(start:finish, predictors{8}(i, start_time:end_time), ...
    'ro--','LineWidth',1.5);
plot(start:finish, y_plot,'x' ...
                              ,'MarkerFaceColor','g'...     
                              ,'LineWidth'      , 2 ...     
                              ,'MarkerSize'     ,10);
                          
hleg1 = legend('uncertainty', 'mean', 'training values');
set(hleg1,'Location','NorthWest');
set(hleg1,'Interpreter','none');
title('Uncertainty Elastic Net lag1');
%% plot Elastic Net lag2 unvertainty
h = [0,0];
h(1) = subplot(4,3,9);

% predictionCRF(i:N:end);
Var = variances{9}(i,start_time:end_time);

fill([x_star; flipdim(x_star,1)], ...
    [(predictors{9}(i, start_time:end_time) + 1.96 * sqrt(Var))'; ...
    flipdim(predictors{9}(i,start_time:end_time) - 1.96*sqrt(Var), 2)'], ...
    [7 7 7] / 8 );
xlabel('Timesteps');
ylabel('Prediction with confidence interval');
hold on;

plot(start:finish, predictors{9}(i, start_time:end_time), ...
    'ro--','LineWidth',1.5);
plot(start:finish, y_plot,'x' ...
                              ,'MarkerFaceColor','g'...     
                              ,'LineWidth'      , 2 ...     
                              ,'MarkerSize'     ,10);
                          
hleg1 = legend('uncertainty', 'mean', 'training values');
set(hleg1,'Location','NorthWest');
set(hleg1,'Interpreter','none');
title('Uncertainty Elastic Net lag2');
%% plot GP lag0 unvertainty
h = [0,0];
h(1) = subplot(4,3,10);

% predictionCRF(i:N:end);
Var = variances{10}(i,start_time:end_time);

fill([x_star; flipdim(x_star,1)], ...
    [(predictors{10}(i, start_time:end_time) + 1.96 * sqrt(Var))'; ...
    flipdim(predictors{10}(i,start_time:end_time) - 1.96*sqrt(Var), 2)'], ...
    [7 7 7] / 8 );
xlabel('Timesteps');
ylabel('Prediction with confidence interval');
hold on;

plot(start:finish, predictors{10}(i, start_time:end_time), ...
    'ro--','LineWidth',1.5);
plot(start:finish, y_plot,'x' ...
                              ,'MarkerFaceColor','g'...     
                              ,'LineWidth'      , 2 ...     
                              ,'MarkerSize'     ,10);
                          
hleg1 = legend('uncertainty', 'mean', 'training values');
set(hleg1,'Location','NorthWest');
set(hleg1,'Interpreter','none');
title('Uncertainty GP lag0');

%% plot GP lag1 unvertainty
h = [0,0];
h(1) = subplot(4,3,11);

% predictionCRF(i:N:end);
Var = variances{11}(i,start_time:end_time);

fill([x_star; flipdim(x_star,1)], ...
    [(predictors{11}(i, start_time:end_time) + 1.96 * sqrt(Var))'; ...
    flipdim(predictors{11}(i,start_time:end_time) - 1.96*sqrt(Var), 2)'], ...
    [7 7 7] / 8 );
xlabel('Timesteps');
ylabel('Prediction with confidence interval');
hold on;

plot(start:finish, predictors{11}(i, start_time:end_time), ...
    'ro--','LineWidth',1.5);
plot(start:finish, y_plot,'x' ...
                              ,'MarkerFaceColor','g'...     
                              ,'LineWidth'      , 2 ...     
                              ,'MarkerSize'     ,10);
                          
hleg1 = legend('uncertainty', 'mean', 'training values');
set(hleg1,'Location','NorthWest');
set(hleg1,'Interpreter','none');
title('Uncertainty GP lag1');

%% plot GP lag2 unvertainty
h = [0,0];
h(1) = subplot(4,3,12);

% predictionCRF(i:N:end);
Var = variances{12}(i,start_time:end_time);

fill([x_star; flipdim(x_star,1)], ...
    [(predictors{12}(i, start_time:end_time) + 1.96 * sqrt(Var))'; ...
    flipdim(predictors{12}(i,start_time:end_time) - 1.96*sqrt(Var), 2)'], ...
    [7 7 7] / 8 );
xlabel('Timesteps');
ylabel('Prediction with confidence interval');
hold on;

plot(start:finish, predictors{12}(i, start_time:end_time), ...
    'ro--','LineWidth',1.5);
plot(start:finish, y_plot,'x' ...
                              ,'MarkerFaceColor','g'...     
                              ,'LineWidth'      , 2 ...     
                              ,'MarkerSize'     ,10);
                          
hleg1 = legend('uncertainty', 'mean', 'training values');
set(hleg1,'Location','NorthWest');
set(hleg1,'Interpreter','none');
title('Uncertainty GP lag2');






end

