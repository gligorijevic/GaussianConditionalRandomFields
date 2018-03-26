function [ ] = timeseriesAnalysis( index, y, x )
N = size(y,1);
T = size(y,2);
figure
plot(1:T, y(index,1:end),'-ro'); 
hold on

f = fft(y,[],2);

% plot(1:T, f(index,1:end),'--g'); 
% hold on
% inverse = ifft(f,[],2);
% plot(1:T, inverse(index,1:end),'.m'); 

% filter = fftfilt(y, 25);
% plot(1:T, filter(index,1:end),'--g'); 

 
window = 7;
coeff24hMA = ones(1, window)/window;

avg24hTempC = filter(coeff24hMA, 1, y, [], 2);
plot(1:T, avg24hTempC(index,1:end), '--g');


title(sprintf('Timeseries plot of %d disease', index));


end

