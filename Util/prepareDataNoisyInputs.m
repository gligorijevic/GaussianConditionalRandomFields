function [ xtrain, lagtrain, utrain, ytrain, xvalid, lagvalid, uvalid, yvalid ] = ...
    prepareDataNoisyInputs(X, y, N, select_features, lag, trainTs, predictTs)
%PREPAREDATA functions simple converts the data into the format ready to
%use for noisy inputs experiments

lagtrain = NaN(N, trainTs, lag);
utrain = NaN(N, trainTs, length(select_features));
xtrain = NaN(N, trainTs, lag+length(select_features));
ytrain = NaN(N, trainTs);

lagvalid = NaN(N, predictTs, lag);
uvalid = NaN(N, predictTs, length(select_features));
xvalid = NaN(N, predictTs, lag+length(select_features));
yvalid = NaN(N, predictTs);

for station_idx = 1:N
    
    X_station = squeeze(X(:,station_idx, select_features));
    Y = y(station_idx, :)';
    
    xtrain_station = [];
    utrain_station = [];
    ytrain_station = [];
    for i = lag+1:lag+trainTs
        xtrain_station = [xtrain_station; Y(i-lag:i-1)'];
        utrain_station = [utrain_station; X_station(i,select_features)];
        ytrain_station = [ytrain_station; Y(i)];
    end
    
    % Build training data (delayed outputs y first)
    lagtrain(station_idx, :, :) = xtrain_station;
    utrain(station_idx, :, :) = utrain_station;
    xtrain(station_idx, :, :) = [xtrain_station utrain_station];
    ytrain(station_idx, :) = ytrain_station;
    
    xvalid_station = [];
    uvalid_station = [];
    yvalid_station = [];
    
    for i = lag+trainTs+1:lag+trainTs+predictTs
        xvalid_station = [xvalid_station; Y(i-lag:i-1)'];
        uvalid_station = [uvalid_station; X_station(i,select_features)];
        yvalid_station = [yvalid_station; Y(i)];
    end
    
    % Build training data (delayed outputs y first)
    lagvalid(station_idx, :, :) = xvalid_station;
    uvalid(station_idx, :, :) = uvalid_station;
    xvalid(station_idx, :, :) = [xvalid_station uvalid_station];
    yvalid(station_idx, :) = yvalid_station; % ? - this i  a good sign!
    
end

end