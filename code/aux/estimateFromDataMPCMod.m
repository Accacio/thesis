function [Aest,Best] = estimateFromDataMPCMod(xhist,M,n,c,o)
  Aest=[];
  Best=[];
  k=size(xhist(:,1:end-1)',1);
  
  FbAxb=[kron(-(M-1),kron(eye(n*c),ones(k,1))) kron((M-1),kron(eye(n*c),ones(k,1)))];
  Faxb=[kron(eye(n*c*M),xhist(:,1:end-1)') circulant(FbAxb,(n*c))];
  Yaxb=reshape(xhist(:,2:end)',1,[])';
%   return
  
  Fsumaij=[kron(ones(1,n*c*M),eye(n*c*M)) zeros(n*c*M,1*n*c*M)];
  Ysumaij=ones(n*c*M,1);  
  
  Fsumb=[zeros(n*c,(n*c*M)^2) kron(ones(1,n*c*M),eye(n*c))];
  Fsumb=[];
  Ysumb=zeros(n*c,1);
  Ysumb=[];
  
  Fa11=[kron(ones((M-1),1),eye((n*c)^2)) (M-1)*eye((M-1)*(n*c)^2)]; 
  Fcontnn1=[circulant(interleave(Fa11,zeros((M-1)*(n*c)^2,(M-1)*(n*c)),n*c),(M*n*c+1)*n*c) zeros(M*(M-1)*(n*c)^2,M*(n*c))];
  
  Ycontnn1Aux=kron(ones(1,M*(M-1)),eye(n*c));
  Ycontnn1=Ycontnn1Aux(:);
  
%   return

  F=[Faxb;    % Y=AX+B
     Fsumaij; % sumaii  = 1
     Fsumb;   % sumb = 0
              % From Structure
     Fcontnn1 % aii  = 1-(n*c*M-1)*amn
  ];
  

  Y=[Yaxb
     Ysumaij    % sum  = 1
     Ysumb       % sumb = 0
     Ycontnn1]; % aii  = 1-(n*c*M-1)*amn
  x=F\Y;
  Aest=reshape(x(1:((M*n*c)^2)),M*n*c,[])';
  
  p=[kron(-(M-1),kron(eye(n*c),1)) kron((M-1),kron(eye(n*c),1))];
  rhoF1surMetrhoF2surM=reshape(x(((M*n*c)^2)+1:end),M*n*c,1);
  Best=circulant(p,(n*c))*rhoF1surMetrhoF2surM;
end
