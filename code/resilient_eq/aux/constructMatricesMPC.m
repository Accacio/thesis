function [A,B] = constructMatricesMPC(alphas,ps,rho)
  alphas;
  ps;
  Maux=size(alphas);
  M=Maux(end);
  nc=size(alphas(:,:,end),1);
  no=size(ps(:,:,end),1);
  A=zeros(M*nc);
  B=zeros(M*nc,1);
  for i=1:M
    for j=1:M
      if i==j
        A((i-1)*nc+1:(i-1)*nc+nc,(j-1)*nc+1:(j-1)*nc+nc)=eye(nc)-(M-1)/M*alphas(:,:,j)*rho;
        B((i-1)*nc+1:(i-1)*nc+nc,1)=B((i-1)*nc+1:(i-1)*nc+nc,1)-(M-1)/M*ps(:,:,j)*rho;
      else
        A((i-1)*nc+1:(i-1)*nc+nc,(j-1)*nc+1:(j-1)*nc+nc)=1/M*alphas(:,:,j)*rho;
        B((i-1)*nc+1:(i-1)*nc+nc,1)=B((i-1)*nc+1:(i-1)*nc+nc,1)+1/M*ps(:,:,j)*rho;
      end
    end
  end
end
