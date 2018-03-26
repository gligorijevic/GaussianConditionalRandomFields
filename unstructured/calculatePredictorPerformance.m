function [ r2,mseP, bias ] = calculatePredictorPerformance( ytrue,prediction )
nanValues=isnan(ytrue);
ytrue(nanValues)=[];
prediction(nanValues)=[];

mseP = mse(prediction-ytrue);
bias=mean(ytrue - prediction);
r2 = 1 - (sum((ytrue - prediction).^2))/sum((ytrue - mean(ytrue)).^2);
end

