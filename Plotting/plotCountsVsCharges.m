function [ ] = plotCountsVsCharges( index, y, x )
N = size(y,1);
T = size(y,2);
figure
hist(1:T, y(index,1:end),'b'); %end-11
hold on
plot(1:T, x(index,1:end),'-bo');
hold on
% plot(1:T, squeeze(x(index,end-1,1:end)),'-go');
title(sprintf('Timeseries plot of %d disease', index));


end

