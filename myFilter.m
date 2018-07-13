function output = myFilter( input )

  output = zeros(size(input));

  for i = 1:size(input, 3)
    for j = 1:size(input, 1)
      
      ts = timeseries( input(j, :, i), 0:(1/256):(255/256) );
      
      f = idealfilter(ts, [.5, 50], 'pass');
      
      output(j, :, i) = f.Data;
      
    end
  end

end

