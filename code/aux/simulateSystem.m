function [xhist,yhist] = simulateSystem(A,B,C,u,xinit)
  K=size(u,2);
  N=size(xinit,1);
  xhist=zeros(N,K+1);
  xhist(:,1)=xinit;
  yhist=[];

  for i=1:K
    xhist(:,i+1)=A*xhist(:,i)+B*u(:,i);
  end
  if ~isempty(C)
    yhist=C*xhist;
  end

end
