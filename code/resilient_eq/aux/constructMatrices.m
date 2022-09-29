function [A,B] = constructMatrices(alphas,ps,rho)

  if size(alphas,2)!=size(ps,2)
    error ("size alpha different from size ps\n")
    return;
  endif
  N=size(alphas,2);
  A=zeros(N);
  B=zeros(N,1);
  for i=1:N
    for j=1:N
      if i==j
        A(i,j)=1-2*(N-1)/N*alphas(j)*rho;
        B(i)=B(i)+2*(N-1)/N*alphas(j)*ps(j)*rho;
      else
        A(i,j)=2/N*alphas(j)*rho;
        B(i)=B(i)-2/N*alphas(j)*ps(j)*rho;
      end
    end
  end
end
