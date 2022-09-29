function [Cmat,Mmat,H,F,G,Qtilde,Rtilde] = getValues(A,B,C,Q,Qbar,R,N)

Cmat=zeros(size(B,1)*N,N*size(B,2));
for i=1:N
    b=zeros(size(B).*[i-1 1]);
    for j=0:N-i
    b=[b;A^(j)*B];
    end
    Cmat(:,i)=b;
end

Mmat=zeros(N*size(A,1),size(A,2));
for i=1:N
Mmat((i-1)*size(A,1)+1:(i-1)*size(A,1)+size(A,1),:)=A^(i);
end

Qtilde=Q;
for i=2:N-1
Qtilde=blkdiag(Qtilde,Q);
end
Qtilde=blkdiag(Qtilde,Qbar);

Rtilde=R;
for i=2:N
Rtilde=blkdiag(Rtilde,R);
end

H=Cmat'*Qtilde*Cmat+Rtilde;
F=Cmat'*Qtilde*Mmat;
G=Mmat'*Qtilde*Mmat+Q;

end