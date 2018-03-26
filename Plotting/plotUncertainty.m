function [] = plotUncertainty(T, steps_ahead, y_plot, predictionCRF, Sigma, predictionCRF_new, Sigma_new, predictionCRF_uf, Sigma_uf, N, i)
start_time = T(1);
end_time = T(2);

x_star = linspace(start_time, end_time + steps_ahead, end_time + steps_ahead);
x_star = x_star(:);


% Defaults for this blog post
width = 5;     % Width in inches
height = 5;    % Height in inches
alw = 0.75;    % AxesLineWidth
fsz = 9;      % Fontsize
lw = 1.5;      % LineWidth
msz = 3;       % MarkerSize

figure

pos = get(gcf, 'Position');
set(gcf, 'Position', [pos(1) pos(2) width*100, height*100]); %<- Set size
set(gca, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties

h = [0,0];
h(1) = subplot(3,1,1);

%% plot gcrf unvertainty
predictionCRF(i:N:end);
Var = Sigma(i:N:end);

fill([x_star; flipdim(x_star,1)], ...
    [(predictionCRF(i:N:end, 1) + 2 * sqrt(Var)); ...
    flipdim(predictionCRF(i:N:end, 1) - 2*sqrt(Var), 1)], ...
    [7 7 7] / 8 );
xlabel('Timesteps');
ylabel('Uncertainty');
hold on;

% plot(start_time:end_time + steps_ahead, predictionCRF(i:N:end, 1), ...
%     'ro--','LineWidth',lw,'MarkerSize'     ,msz);
% plot(start_time:end_time + steps_ahead, y_plot(i:N:end),'b--' ...
%                               ,'MarkerFaceColor','g'...     
%                               ,'LineWidth'      , lw ...     
%                               ,'MarkerSize'     ,msz);
plot(start_time:end_time + steps_ahead, predictionCRF(i:N:end, 1), 'b-',start_time:end_time + steps_ahead,...
    y_plot(i:N:end), 'r--', 'LineWidth',lw, 'MarkerSize',msz); %<- Specify plot properites
axis([0 25 -0.5 2])                          
% hleg1 = legend('uncertainty', 'mean', 'training values');
% set(hleg1,'Location','SouthWest');
% set(hleg1,'Interpreter','none');

title('GCRF');

%% plot gcrf_new uncertainty
h(2) = subplot(3,1,2);
Var_new = Sigma_new(i:N:end);

fill([x_star; flipdim(x_star,1)], ...
    [(predictionCRF_new(i:N:end) + 2 * sqrt(Var_new)); ...
    flipdim(predictionCRF_new(i:N:end) - 2 * sqrt(Var_new), 1)], ...
    [7 7 7] / 9 );
xlabel('Timesteps')
ylabel('Uncertainty')
hold on;

% plot(start_time:end_time + steps_ahead, predictionCRF_new(i:N:end), ...
%     'ro--','LineWidth',1);
% plot(start_time:end_time + steps_ahead, y_plot(i:N:end),'b--'...
%                               , 'MarkerFaceColor','g'...          
%                               ,'LineWidth'      , 1 ...     
%                               ,'MarkerSize'     ,7);
plot(start_time:end_time + steps_ahead, predictionCRF_new(i:N:end), 'b-',start_time:end_time + steps_ahead,...
    y_plot(i:N:end), 'r--', 'LineWidth',lw, 'MarkerSize',msz); %<- Specify plot properites
axis([0 25 -0.5 2])   
% hleg2 = legend('uncertainty','mean','training values');
% set(hleg2,'Location','SouthWest');
% set(hleg2,'Interpreter','none');
title('uGCRF');

%% plot gcrf_uf uncertainty
h(3) = subplot(3,1,3);

Var_uf = Sigma_uf(i:N:end);

fill([x_star; flipdim(x_star,1)], ...
    [(predictionCRF_uf(i:N:end) + 2 * sqrt(Var_uf)); ...
    flipdim(predictionCRF_uf(i:N:end) - 2 * sqrt(Var_uf), 1)], ...
    [7 7 7] / 9 );
xlabel('Timesteps')
ylabel('Uncertainty')
hold on;

% plot(start_time:end_time + steps_ahead, predictionCRF_uf(i:N:end), ...
%     'ro--','LineWidth',1);
% plot(start_time:end_time + steps_ahead, y_plot(i:N:end),'b--'...
%                               , 'MarkerFaceColor','g'...          
%                               ,'LineWidth'      , 1 ...     
%                               ,'MarkerSize'     ,7);
plot(start_time:end_time + steps_ahead, predictionCRF_uf(i:N:end), 'b-',start_time:end_time + steps_ahead,...
    y_plot(i:N:end), 'r--', 'LineWidth',lw, 'MarkerSize',msz); %<- Specify plot properites
axis([0 25 -0.5 2])   
hleg2 = legend('uncertainty','mean','training values');
set(hleg2,'Location','SouthWest');
set(hleg2,'Interpreter','none');
title('ufGCRF');

linkaxes(h);

% Set Tick Marks
% set(gca,'XTick',0:5:25);
% set(gca,'YTick',-0.5:1:2);

% Here we preserve the size of the image when we save it.
set(gcf,'InvertHardcopy','on');
set(gcf,'PaperUnits', 'inches');
papersize = get(gcf, 'PaperSize');
left = (papersize(1)- width)/2;
bottom = (papersize(2)- height)/2;
myfiguresize = [left, bottom, width, height];
set(gcf,'PaperPosition', myfiguresize);

% Save the file as PNG
print('pred_unc','-dpng','-r300');


hold off;


