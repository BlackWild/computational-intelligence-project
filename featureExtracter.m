function features = featureExtracter( data )
  
  NUM_OF_FEATURES = 2;
  [~, ~, numOfExperiments] = size(data);
  
  features = zeros(NUM_OF_FEATURES, numOfExperiments);
  
  for i = 1:numOfExperiments
    
    features(1, i) = max(abs(data(1, :, i)));
    features(2, i) = mean(data(1, :, i));
    
  end

end

