function [ output ] = normalizer( input, mu, sigma )

  [numOfFeatures, ~] = size(input);
  output = zeros(size(input));
  
  for i = numOfFeatures
    
    output(i, :) = (input(i, :) - mu(i))/sigma(i);
    
  end

end

