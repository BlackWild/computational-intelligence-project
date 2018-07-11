function J = jComputer( features, labels )

  [numOfFeatures, ~] = size(features);
  J = zeros(numOfFeatures, 1);
  
  for i = 1:numOfFeatures
    
    temp = features(i, :);
    
    mt = mean(temp             );
    m0 = mean(temp(labels == 0));
    m1 = mean(temp(labels == 1));
    
    s0 = var(temp(labels == 0));
    s1 = var(temp(labels == 1));
    
    J(i) = ( (m0-mt)^2 + (m1-mt)^2 )/( s0 + s1 );
    
  end

end

