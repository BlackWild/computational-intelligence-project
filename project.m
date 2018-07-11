
%% Loading Data
load('./Data/SubHandData');

%% Extracting Features
[features, NUM_OF_FEATURES] = featureExtracter(TrainX);
mu = mean(features, 2);
sigma = std(features, 0, 2);

features = normalizer(features, mu, sigma);

%% Choosing Best Features

J = jComputer(features, Trainy);

[sortedJ, sortedJindex] = sort(J, 'descend');

% bFeatures = features(sortedJindex(1:3), :);
bFeatures = features( J > .01 , :);


%% Estimating Error





























































































