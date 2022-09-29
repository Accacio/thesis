function F = FsymConst(n,acc)

if acc==1
  F=[];
  return
end
myeye=eye(n-1);
myeye=myeye(1:acc-1,:);
myinterleave=interleave(-1*eye(acc-1),zeros(acc-1,n-1),1);
myinterleave=myinterleave(:,1:(acc-1)*n-(n-acc));
zeros(acc-1,(n-acc)*n); 
zeros(acc-1,n-acc+1);

F=[zeros(acc-1,(n-acc)*n) zeros(acc-1,n-acc+1) myeye myinterleave;FsymConst(n,acc-1)];


end