function [Aest,Hest] = estimateFromDataMPC(thetahist,lambdahist,M,n,alpha,ktotal)
  Aest=[];
  Best=[];
  Hest=zeros(n,n,M);
  Hblockest=[];
  I_M_N=kron(ones(M,1),eye(n));
  for unit=1:M
    theta_unit=thetahist((unit-1)*(n)+1:(unit-1)*(n)+1+n-1,:);
    lambda_unit=reshape(lambdahist(:,unit,:),n,[]);
    
    theta_unit=theta_unit(:,1:ktotal);
    lambda_unit=lambda_unit(:,1:ktotal);

    theta_unit_k=theta_unit(:,1:end-2);
    theta_unit_k_1=theta_unit(:,2:end-1);
    lambda_unit_k=lambda_unit(:,1:end-2);
    lambda_unit_k_1=lambda_unit(:,2:end-1);

    F=[kron(eye(n),(theta_unit_k_1-theta_unit_k)');FsymConst(n,n)];

    y=[reshape((lambda_unit_k_1-lambda_unit_k)',1,[])';zeros((n+1)*n/2-n,1)];

    h_unit_flat=-F\y;

    Hest(:,:,unit)=reshape(h_unit_flat,n,n);
    Hblockest=blkdiag(Hblockest,Hest(:,:,unit));
  end

  Aest=eye(M*n)-alpha*(eye(M*n)-I_M_N/(I_M_N'*I_M_N)*I_M_N')*(Hblockest);

  return
end
