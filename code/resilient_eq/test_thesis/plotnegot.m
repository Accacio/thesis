function h = plotnegot(theta,lambda,K)
  K=ceil(K);
  dtheta(:,:,:)=theta(:,1:size(theta,2),K,:);
  subplot(2,1,1);
  plot(1:size(theta,2),reshape(permute(dtheta,[ 1 3 2 ]),[size(theta,1)*size(theta,4),size(theta,2)]));
  title(['Theta for Negotiation for K=' num2str(K) ])
  xlim([1 size(theta,2)])
  subplot(2,1,2);
  dlambda(:,:,:)=lambda(:,1:size(lambda,2),K,:);
  h=plot(1:size(lambda,2),reshape(permute(dlambda,[ 1 3 2 ]),[size(lambda,1)*size(lambda,4),size(lambda,2)]));
  title(['lambda Negotiation for K=' num2str(K) ])
end