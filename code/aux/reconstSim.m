function S = reconstSim(index,n)

S=zeros(n);
for i=1:n
  for j=1:n
    i
    j
    if i~=j
      S(i,j)=index(i*j-1)
    end
      S(i,j)=index(1+n*)
  end
end
return