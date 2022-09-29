% [[file:../../../docs/org/decomposition_methods/secureDMPC.org::onlineSecureEstimateHFRandWaitConvergence][onlineSecureEstimateHFRandWaitConvergence]]
function [coordinator, subsystems, theta, lambdaHist] = onlineSecureEstimateHFRandWaitConvergence(coordinator, subsystems, lambdaHist, theta, K, negotP, rho, Ac, bc, proj)
  n = subsystems.n;
  mask=triu(true(n));
  forget=0.1;
  % estHorz=n/2*(n+1)+n;
  estHorz=n/2*(n+1)+n;

 if K==1
   coordinator.hest = zeros(n/2*(n+1)+n,negotP,coordinator.simK,subsystems.M);
 end
% onlineSecureEstimateHFRandWaitConvergence ends here

% [[file:../../../docs/org/decomposition_methods/secureDMPC.org::*symmetric by definition][symmetric by definition:11]]
  %% estimation

  for i=1:subsystems.M
    estIt = 1;
    err=2000*ones(n/2*(n+1)+n,1);

    while norm(err)>1e-12
      thetap = rand(n,subsystems.M);
      tproj = reshape(proj(thetap(:),Ac,bc),[subsystems.n,subsystems.M]);
      theta(:,estIt,K,i) = tproj(:,i);

      [ subsystems, lambdap ] = getLambdas(subsystems,tproj,estIt,K);

      thetap = tproj(:,i);

      if estIt==1
        if K==1
          hest_1 = zeros(n/2*(n+1)+n,1);
        else
          hest_1 = coordinator.hest(:,estIt,K,i);
        end
        coordinator.hest(:,estIt,K,i) = hest_1;
        F_1 = 100*eye(n/2*(n+1)+n);
      else
        hest_1 = coordinator.hest(:,estIt-1,K,i);
        F_1 = coordinator.F(:,:,estIt-1,K,i);
      end

      b=theta(:,estIt,K,i);
      upsilon=lambdap(:,i);
      [hest, F, epsilon  ] = estimateHFrls(hest_1, F_1, b, upsilon, forget);
      coordinator.hest(:,estIt,K,i) = hest;
      coordinator.F(:,:,estIt,K,i) = F;
      coordinator.epsilon(:,estIt,K,i) = epsilon;
      H=zeros(n);
      H(mask') = -hest(1:n/2*(n+1));
      coordinator.Hest(:,:,estIt,K,i) = H'+H-diag(diag(H));
      fest = -hest(n/2*(n+1)+1:end);
      coordinator.fest(:,estIt,K,i) = fest;
      if estIt > 1
        err=coordinator.hest(:,estIt,K,i)-coordinator.hest(:,estIt-1,K,i);
      end
      coordinator.HFestErr(:,estIt,K,i)=err;
%       norm(err)
      estIt=estIt+1;
    end

    coordinator.HFinalEst(:,:,K,i) = coordinator.Hest(:,:,estIt-1,K,i);
    coordinator.estFinalIt(:,K,i)=estIt-1;
    estIt;

  end
% symmetric by definition:11 ends here

% [[file:../../../docs/org/decomposition_methods/secureDMPC.org::*symmetric by definition][symmetric by definition:12]]
  %% negotiation
  negot=estIt-1
  err=2e3;

  while norm(err)>1e-12

    thetap = theta(:,negot,K,:);

    [ subsystems, lambdap ] = getLambdas(subsystems,thetap,negot,K);

    lambdaHist(:,negot,K,:) = lambdap;

    if K>1
      for i=1:subsystems.M
        errComp1 = coordinator.hest(1:n/2*(n+1),coordinator.estFinalIt(:,K,i),K,i)-coordinator.hest(1:n/2*(n+1),coordinator.estFinalIt(:,K,i),1,i);
        coordinator.errComp1(:,negot,K,i)=errComp1;
        if norm(errComp1)>1e-4
          % estimate $T^{-1}$ using hest = T*h_orig
          invT = coordinator.Hest(:,:,estHorz,1,i)*inv(coordinator.Hest(:,:,estHorz,K,i));
          % lambdap(:,i)=inv(T)*lambdap(:,i);
          lambdap(:,i)=-coordinator.Hest(:,:,estHorz,1,i)*theta(:,negot,K,i)-invT*coordinator.fest(:,coordinator.estFinalIt(:,K,i),K,i);
        end
      end
    end

    lambdaHist(:,negot,K,:) = lambdap;
    theta(:,negot+1,K,:) = theta(:,negot,K,:)+rho*(lambdaHist(:,negot,K,:) - mean(lambdap,2));

    % norm(err)
    negot=negot+1;
  end
end
% symmetric by definition:12 ends here
