%*****************************************************************
% Center for Information Science and Technology
% Temple University
% Philadelphia, PA 19122
%*****************************************************************

%*****************************************************************
% REGRESSION PROBLEM - PREDICT REAL VALUE OF THE OUTPUT
%*****************************************************************

% load the data into a matrix (last column represents target variable) 
load housing.txt
data = housing;
clear housing;

% print out matrix dimensions and set some variables
[n_examples, n_cols] = size(data);
n_features = n_cols - 1;

% First look at the data:

% look at the histograms of some features and the target variable
hist(data(:, 1), 30);  % feature No. 1 (continuous valued feature)
hist(data(:, 4));      % feature No. 4 (binary valued feature)
hist(data(:, 14), 30); % target variable
% check the mean and standard deviation
mean(data)
std(data)
% look at how some features are correlated
% among one another and with the target variable
plotmatrix(data(:, [4 6 13 14]))
% compute correlation coefficient (feature No. 4 and target variable)
corrcoef(data(:, [4 14]))
corrcoef(data(:, [6 14]))
corrcoef(data(:, [13 14]))

mse_total=ones(30,1);
r_sq_total=ones(30,1);

% Prepare for training:
for iter=1:30
% split the data into training, validation and test subsets
% leave 30% of data for testing of our model (in the testset)
[tr_val, test] = divideset(data, 30);

% normalize all the features to mean zero and variance 1;
% note that target variable do not have to be normalized
% because of the type of neural networks used later on
%%%%%%%NORMALIZATION%%%%%%%%%%%%%%%%%%%[meanv, stdv, tr_val(:, 1 : n_features)] = normalize(tr_val(:, 1 : n_features), [], []);
[meanv, stdv, tr_val(:, 1 : n_features)] = normalize(tr_val(:, 1 : n_features), [], []);


% assign 30% of tr_val to validation and 70% to training set
[tr, val] = divideset(tr_val, 30);                                         

% Set neural network parameters
info.hidd = 5;			% number of hidden neurons
info.epochs = 100;	% max number of training iterations (epochs)
info.show = 10;		% show training results each 'show' epochs
info.max_fail = 5;	% if error does not decrease on 'val' set in 
                     % 'max_fail' consecutive epochs, stop the training
                     
% Train the neural network
[trash, net]=neural_simple(tr,val,[],info);

% Predict target variable on the test set
%
% first, normalize test data the same way tr_val was normalized
%%%%%%%NORMALIZATION%%%%%%%%%%%%%%%%%%%[meanv, stdv, test(:, 1 : n_features)] = normalize(test(:, 1 : n_features), meanv, stdv);
%[meanv, stdv, test(:, 1 : n_features)] = normalize(test(:, 1 : n_features), meanv, stdv);


prediction = sim(net, test(:, 1:n_features)')';


%pred_train=sim(net, tr(:, 1:n_features)')';%
%pred_val=sim(net, val(:, 1:n_features)')';%


% calculate Mean Square Error (mse)
pred_error = prediction - test(:, n_features + 1);


%pred_error_tr = pred_train - tr(:, n_features + 1);
%pred_error_val = pred_val - val(:, n_features + 1);


mse = (pred_error' * pred_error) / length(pred_error);
mse_total(iter,1)=mse;


%mse_tr = (pred_error_tr' * pred_error_tr) / length(pred_error_tr) %
%mse_val = (pred_error_val' * pred_error_val) / length(pred_error_val)%

% Plot the histogram of prediction errors
hist(pred_error, 30);
%hist(pred_error_tr, 30);%
%hist(pred_error_val, 30);%

% calculate R-square value
R_square = 1 - mse / std(test(:, n_features + 1)) ^ 2;
r_sq_total(iter,1)=R_square;
%R_square_tr = 1 - mse_tr / std(tr(:, n_features + 1)) ^ 2
%R_square_val = 1 - mse_val / std(val(:, n_features + 1)) ^ 2

end;
average_R_square= mean(r_sq_total);
average_MSE=mean(mse_total);
std_R_square=std(r_sq_total);
std_MSE=std(mse_total);
r_resultsss=zeros(2,2);
r_resultsss(1,1)=average_MSE;
r_resultsss(1,2)=average_R_square;
r_resultsss(2,1)=std_MSE;
r_resultsss(2,2)=std_R_square;

return;

