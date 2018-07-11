function [ output ] = normalizer( input, mu, sigma )

  [numOfFeatures, ~] = size(input);
  output = zeros(size(input));
  
  for i = 1:numOfFeatures
    
    output(i, :) = (input(i, :) - mu(i))/sigma(i);
    
  end

end

