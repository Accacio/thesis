% [[file:../../../docs/org/decomposition_methods/secureDMPC.org::onlineSecureEstimateHRand][onlineSecureEstimateHRand]]
function [coordinator, subsystems, theta, lambdaHist] = onlineSecureEstimateHRand(coordinator, subsystems, lambdaHist, theta, K, negotP, rho)
  n = subsystems.n;
  mask=triu(true(n));
  forget=0.1;
  % estHorz=n/2*(n+1)+n;
  estHorz=n/2*(n+1)+n;
  for p=1:negotP

    if p==1 & K==1
      coordinator.epsilon = zeros(n,negotP,coordinator.simK,subsystems.M);
      coordinator.F = zeros(n/2*(n+1),n/2*(n+1),negotP,coordinator.simK,subsystems.M); % must be symmetric
      coordinator.hest = zeros(n/2*(n+1),negotP,coordinator.simK,subsystems.M);
    end

    if p<=estHorz
      thetap=rand(n,subsystems.M);
      thetap=thetap./(sum(thetap,2));
      t(:,:)=theta(1,1,1,:);
      theta(:,p,K,:) = sum(t)*thetap;
    end
    thetap = theta(:,p,K,:);

    [ subsystems, lambdap ] = getLambdas(subsystems,thetap,p,K);
    lambdaHist(:,p,K,:) = lambdap;

    if p<=estHorz
      for i=1:subsystems.M
        if p==1
          if K==1
            hest_1 = zeros(n/2*(n+1),1);
            coordinator.hest(:,p,K,i)=hest_1;
            F_1 = 100*eye(n/2*(n+1));
          else
            hest_1 = coordinator.hest(:,p,K,i);
            coordinator.hest(:,p,K,i)=hest_1;
            F_1 = 100*eye(n/2*(n+1));
          end
        else
          hest_1 = coordinator.hest(:,p-1,K,i);
          F_1 = coordinator.F(:,:,p-1,K,i);
        end
        b=theta(:,p,K,i);
        upsilon=lambdaHist(:,p,K,i);
        [hest, F, epsilon  ] = estimateHrls(hest_1, F_1, b, upsilon, forget);
        coordinator.hest(:,p,K,i) = hest;
        coordinator.F(:,:,p,K,i) = F;
        coordinator.epsilon(:,p,K,i) = epsilon;
        H=zeros(n);
        H(mask') = -hest(1:n/2*(n+1));
        coordinator.Hest(:,:,p,K,i) = H'+H-diag(diag(H));
        coordinator.eigHest(:,K,i) = eig(coordinator.Hest(:,:,p,K,i));
      end
    end

    if p>=estHorz
      theta(:,p+1,K,:) = theta(:,p,K,:)+rho*(lambdaHist(:,p,K,:) - mean(lambdap,2));
    end

  end
end
% onlineSecureEstimateHRand ends here
