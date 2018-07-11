function [features, NUM_OF_FEATURES] = featureExtracter( data )
  
  NUM_OF_FEATURES = 6;
  [~, ~, numOfExperiments] = size(data);
  
  features = zeros(NUM_OF_FEATURES, numOfExperiments);
  
  for i = 1:numOfExperiments
    
    [~, t] = max(abs(data(2, :, i)));
    features(1, i) = t;
    
    t = cov(data(2, :, i), data(4, :, i));
    features(2, i) = t(1,1);
    features(3, i) = t(2,2);
    features(4, i) = t(1,2);
    
    features(5, i) = mean(mean(data(:, :, i)));
    features(6, i) = max(var(data(:, :, i)));
    
  end

end

