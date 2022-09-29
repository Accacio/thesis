function [Aest,Best] = estimateFromData(xhist)
  N=size(xhist(:,1),1);
  k=size(xhist(:,1:end-1)',1);
  Faxb=[kron(eye(N),xhist(:,1:end-1)') kron(eye(N),ones(k,1))];
  Fsumaii=[kron(ones(1,N),eye(N)) zeros(N)];
  Fsumb=[zeros(1,N^2) ones(1,N)];
  Fa11=[ones(N-1,1) (N-1)*eye(N-1)];
  Fcontnn1=[circulant(interleave(Fa11,zeros(N-1),1),N+1) zeros(N*(N-1),N)];
  F=[Faxb;    % Y=AX+B
     Fsumaii; % sumaii  = 1
     Fsumb;   % sumb = 0
              % From Structure
     Fcontnn1 % aii  = 1-(N-1)*amn
  ];

  Yaxb=reshape(xhist(:,2:end)',1,[])';
  Ysumaii=ones(1,N);
  Ysumb=0;
  Ycontnn1=ones(1,N*(N-1));
  Y=[Yaxb
     Ysumaii'    % sum  = 1
     0           % sumb = 0
     Ycontnn1']; % aii  = 1-(N-1)*amn
  x=F\Y;

  Aest=reshape(x(1:end-N),N,[])';
  Best=[x(end-N+1:end)];
end
