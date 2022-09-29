function [Mmat,Cmat] = getMPCValuesY(A,B,C,N)
  p=size(A,1);
  c=size(B,2);
  o=size(C,1);
  Cmat=zeros(size(C,1)*N,N*size(B,2));
  Mmat=zeros(N*o,p);

  for i=1:N
    Mmat((i-1)*o+1:(i-1)*o+o,:)=C*A^(i);
    b=zeros(size(C,1).*(i-1), size(B,2));
    for j=0:N-i
      b=[b;C*A^(j)*B];
    end
    Cmat(:,c*(i-1)+1:c*i)=b;
  end
end
