function f = fitness( indexes )

  load ./Output/forFitnessFunction;
  
  bFeatures = features(indexes == 1, :);
  
  t0 = bFeatures(:, Trainy == 0);
  t1 = bFeatures(:, Trainy == 1);
  
  ut = mean(bFeatures, 2);
  u0 = mean(t0, 2);
  u1 = mean(t1, 2);
  
  n0 = size(t0, 2);
  n1 = size(t1, 2);
  
  s0 = 0;
  s1 = 0;
  
  for i = 1:n0
    
    a = t0(:, i) - u0;
    s0 = s0 + (a' * a)/n0;
 
  end
  
  for i = 1:n1
    
    a = t1(:, i) - u1;
    s1 = s1 + (a' * a)/n1;
    
  end
  
  a = u0 - ut;
  b = u1 - ut;
  sb = a' * a + b' * b;
  
  J = sb/(s0 + s1);
  
  f = -J;
  

end

