
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

%%

bestMLPN1 = -1;
bestMLPNet1 = -1;
bestMLPMeanError1 = 100;
for n = 20:30

  meanError = 0;
  for k = 1:5
    validations = bFeatures(:, 1+(k-1)*36:k*36);
    validationLabels = Trainy(1+(k-1)*36:k*36);
    trains = [bFeatures(:, 1:(k-1)*36), bFeatures(:, 1+k*36:end)];
    trainLabels = [Trainy(1:(k-1)*36), Trainy(1+k*36:end)];

    net = patternnet(n);
    net = train(net, trains, trainLabels);

    estVals = net(validations) >= 0.5;

    error = sum(estVals ~= validationLabels);
    meanError = meanError + (error);
  end
  
  meanError = meanError / 180 * 100;

  if meanError < bestMLPMeanError1
    bestMLPN1 = n;
    bestMLPNet1 = net;
    bestMLPMeanError1 = meanError;
  end
  
end
clear n k net estVals error meanError;
clear validations validationLabels trains trainLabels;



%%

bestRBFN1 = -1;
bestRBFR1 = -1;
bestRBFNet1 = -1;
bestRBFMeanError1 = 100;
for n = 20:2:30
for r = [.2 .4 .7]

  meanError = 0;
  for k = 1:5
    validations = bFeatures(:, 1+(k-1)*36:k*36);
    validationLabels = Trainy(1+(k-1)*36:k*36);
    trains = [bFeatures(:, 1:(k-1)*36), bFeatures(:, 1+k*36:end)];
    trainLabels = [Trainy(1:(k-1)*36), Trainy(1+k*36:end)];

    net = newrb(trains, trainLabels, 0, r, n, n);

    estVals = net(validations) >= 0.5;

    error = sum(estVals ~= validationLabels);
    meanError = meanError + (error);
  end
  
  meanError = meanError / 180 * 100;

  if meanError < bestRBFMeanError1
    bestRBFN1 = n;
    bestRBFR1 = r;
    bestRBFNet1 = net;
    bestRBFMeanError1 = meanError;
  end
  
end
end
clear n k r net estVals error meanError;
clear validations validationLabels trains trainLabels;






















































































