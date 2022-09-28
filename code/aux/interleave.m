function [Y] = interleave(A,B,n)
  Y=[];
  for i=1:n:size(A,2)
    Y=[Y A(:,i:i+n-1) B];
  end
end
