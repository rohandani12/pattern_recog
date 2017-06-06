%--------------------Load data---------------------------
clc;
clear all;
data = load('train_sp2017_v19'); 

%------------Random initial centroids-------------

i=0;k=5;

p = randperm(size(data,1));
for i = 1:k
    centr(i,:) = data(p(i),:);
end

data_dim = length(data(1,:));
num_data   = length(data(:,1));
old_centres = centr;
diff = 1.;
while diff > 0.0
  assign = [];
%----------------Assign data to closest centroid-----------------

  for d = 1 : length( data(:, 1) );

    %min_diff = sum(( data( d, :) - centr( 1,:) ).^2);
    min_diff = sum(abs( data( d, :) - centr( 1,:) ));
    curr_assign = 1;

    for c = 2 : k;
      %dist2c = sum(( data( d, :) - centr( c,:) ).^2);
      dist2c = sum(abs( data( d, :) - centr( c,:) ));
      if( min_diff >= dist2c)
        curr_assign = c;
        min_diff = dist2c;
      end
    end

    %----------Assign next datapoint-----------
    assign = [ assign; curr_assign];
  end
  old_centrs = centr;

  %-----------------Recalculate centroids-------------------
  
  centr = zeros(k, data_dim);
  cluster_pts = zeros(k, 1);

  for d = 1: length(assign);
    centr( assign(d),:) = centr( assign(d),:) + data(d,:);
    cluster_pts( assign(d), 1 ) = cluster_pts( assign(d), 1 ) + 1;
  end

  for c = 1: k;
    if( cluster_pts(c, 1) ~= 0)
      centr( c , : ) = centr( c, : ) / cluster_pts(c, 1);
    else
      %----if no pts in cluster, random initialization---
      centr( c , : ) = rand( 1, data_dim);
    end
  end

  %---------Check difference in positions ----------------
  diff = sum (sum( (centr - old_centrs).^2 ) );

end