function [features, NUM_OF_FEATURES] = featureExtracter( data )
  
  [~, ~, numOfExperiments] = size(data);
  
  features = zeros(1000, numOfExperiments); 
  c = 0;
 
  for j = 1:30

    % % % Statistical

    c = c + 1 + 1 + 1;
    for i = 1:numOfExperiments
      
      a = data(j, :, i);
      
      features(c,   i) = sum( a.^2 );     % Energy
      
      [~, t] = max(abs(a));
      features(c-1, i) = t;               % Peak time
      
      features(c-2, i) = max(a) - min(a); % Range
      
    end

    for k = j:30
      c = c + 1;
      for i = 1: numOfExperiments
        t = cov(data(j, :, i), data(k, :, i));
        features(c, i) = t(1,2);     % Covariance + Variance
      end
    end
    


    % Frequency
    
    c = c + 1+1+1+1+1+1+1;
    for i = 1:numOfExperiments
      
      a = data(j, :, i);
      
      features(c,   i) = meanfreq(a, 256);
      features(c-1, i) = medfreq(a, 256);
      
      features(c-2, i) = bandpower(a, 256, [2, 8]);
      features(c-3, i) = bandpower(a, 256, [8, 15]);
      features(c-4, i) = bandpower(a, 256, [15, 22]);
      features(c-5, i) = bandpower(a, 256, [22, 29]);
      features(c-6, i) = bandpower(a, 256, [29, 36]);
      
    end


  end
    
  
  NUM_OF_FEATURES = c;
  features = features(1:c, :);

end

