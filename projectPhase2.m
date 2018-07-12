
%% Loading Data
load ./Data/SubHandData ;

%% Extracting Features
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
  'Generations', 150                           ...
);

bestIndexes = ga(@fitness,NUM_OF_FEATURES,[],[],[],[],[],[],[],[],options);

bFeatures = features(bestIndexes == 1, :);
bFeatures_test = features_test(bestIndexes == 1, :);

%% Estimating Error

%%

bestMLPN1 = -1;
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
    bestMLPMeanError1 = meanError;
  end
  
end
clear n k net estVals error meanError;
clear validations validationLabels trains trainLabels;

%%

bestRBFN1 = -1;
bestRBFR1 = -1;
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
    bestRBFMeanError1 = meanError;
  end
  
end
end
clear n k r net estVals error meanError;
clear validations validationLabels trains trainLabels;


%% Best nets

bestMPLnet1 = patternnet(bestMLPN1);
bestMPLnet1 = train(bestMPLnet1, bFeatures, Trainy);

bestRBFnet1 = newrb(bFeatures, Trainy, 0, bestRBFR1, bestRBFN1, bestRBFN1);

estimatedLabels_MLP1 = ( bestMPLnet1(bFeatures_test) >= 0.5 );
estimatedLabels_RBF1 = ( bestRBFnet1(bFeatures_test) >= 0.5 );

save ./Output/SecondPhaseEstimations ...
  estimatedLabels_MLP1 estimatedLabels_RBF1 ;



