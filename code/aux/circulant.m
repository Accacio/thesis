function [A] = circulant(orig,N)
  A=[orig];
  for i=1:floor((size(orig,2)-1)/N)
  A=[A; circshift(orig,(N)*i,2)];
  end
end
