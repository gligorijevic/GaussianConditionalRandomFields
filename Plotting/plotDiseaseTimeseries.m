function [  ] = plotDiseaseTimeseries( index, y, x )
N = size(y,1);
T = size(y,2);
figure
plot(1:T, y(index,1:end),'-ro'); %end-11
hold on
plot(1:T, squeeze(x(index,end-2,1:end)),'-bo');
hold on
plot(1:T, squeeze(x(index,end-1,1:end)),'-go');
title(sprintf('Timeseries plot of %d disease', index));

end

