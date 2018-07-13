
%% Loading Data
load ./Data/SubHandData ;

%% Extracting Features

load ./Output/FilteredData;

[features, NUM_OF_FEATURES] = featureExtracter(TrainX);
mu = mean(features, 2);
sigma = std(features, 0, 2);

features = normalizer(features, mu, sigma);

features_test = featureExtracter(TestX);
features_test = normalizer(features_test, mu, sigma);

save ./Output/forFitnessFunction features Trainy;

%% Choosing Best Features (GA)

options = gaoptimset(                          ...
  'PopulationType', 'bitstring',               ...
  'PlotFcn', {@gaplotbestf, @gaplotbestindiv}, ...
  'Generations', 1000                          ...
);

bestIndexes = ga(@fitness,NUM_OF_FEATURES,[],[],[],[],[],[],[],[],options);

bFeatures = features(bestIndexes == 1, :);
bFeatures_test = features_test(bestIndexes == 1, :);

%% Estimating Error

%%

bestMLPN2 = -1;
bestMLPMeanError2 = 100;
for n = 25:40

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

  if meanError < bestMLPMeanError2
    bestMLPN2 = n;
    bestMLPMeanError2 = meanError;
  end
  
end
clear n k net estVals error meanError;
clear validations validationLabels trains trainLabels;

%%

bestRBFN2 = -1;
bestRBFR2 = -1;
bestRBFMeanError2 = 100;
for n = 20:2:30
for r = [1 1.1 1.2 1.3 1.4 1.5]

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

  if meanError < bestRBFMeanError2
    bestRBFN2 = n;
    bestRBFR2 = r;
    bestRBFMeanError2 = meanError;
  end
  
end
end
clear n k r net estVals error meanError;
clear validations validationLabels trains trainLabels;


%% Best nets

bestMPLnet2 = patternnet(bestMLPN2);
bestMPLnet2 = train(bestMPLnet2, bFeatures, Trainy);

bestRBFnet2 = newrb(bFeatures, Trainy, 0, bestRBFR2, bestRBFN2, bestRBFN2);

estimatedLabels_MLP2 = ( bestMPLnet2(bFeatures_test) >= 0.5 );
estimatedLabels_RBF2 = ( bestRBFnet2(bFeatures_test) >= 0.5 );

save ./Output/SecondPhaseEstimations ...
  estimatedLabels_MLP2 estimatedLabels_RBF2 ;



