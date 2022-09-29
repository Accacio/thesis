% [[file:../../../docs/org/decomposition_methods/secureDMPC.org::onlineSecureEstimateHFRand][onlineSecureEstimateHFRand]]
function [coordinator, subsystems, theta, lambdaHist] = onlineSecureEstimateHFRand(coordinator, subsystems, lambdaHist, theta, K, negotP, rho, Ac, bc, proj)
  n = subsystems.n;
  mask=triu(true(n));
  forget=0.1;
  % estHorz=n/2*(n+1)+n;
  estHorz=n/2*(n+1)+n;
  for p=1:negotP

    if p==1 & K==1
      coordinator.epsilon = zeros(n,negotP,coordinator.simK,subsystems.M);
      coordinator.F = zeros(n/2*(n+1)+n,n/2*(n+1)+n,negotP,coordinator.simK,subsystems.M); % must be symmetric
      coordinator.hest = zeros(n/2*(n+1)+n,negotP,coordinator.simK,subsystems.M);
    end

    if p<=estHorz
      thetap = rand(n,subsystems.M);
      tproj = reshape(proj(thetap(:),Ac,bc),[subsystems.n,subsystems.M]);
      theta(:,p,K,:) = tproj;
      % thetap=thetap./(sum(thetap,2));
      % t(:,:)=theta(1,1,1,:);
      % theta(:,p,K,:) = sum(t)*thetap;
    end
    thetap = theta(:,p,K,:);

    [ subsystems, lambdap ] = getLambdas(subsystems,thetap,p,K);

    if p<=estHorz
      for i=1:subsystems.M
        if p==1
          if K==1
            hest_1 = zeros(n/2*(n+1)+n,1);
          else
            hest_1 = coordinator.hest(:,p,K,i);
          end
          coordinator.hest(:,p,K,i)=hest_1;
          F_1 = 100*eye(n/2*(n+1)+n);
        else
          hest_1 = coordinator.hest(:,p-1,K,i);
          F_1 = coordinator.F(:,:,p-1,K,i);
        end
        b=theta(:,p,K,i);
        upsilon=lambdap(:,i);
        [hest, F, epsilon  ] = estimateHFrls(hest_1, F_1, b, upsilon, forget);
        coordinator.hest(:,p,K,i) = hest;
        coordinator.F(:,:,p,K,i) = F;
        coordinator.epsilon(:,p,K,i) = epsilon;
        H=zeros(n);
        H(mask') = -hest(1:n/2*(n+1));
        coordinator.Hest(:,:,p,K,i) = H'+H-diag(diag(H));
        fest = -hest(estHorz-n+1:end);
        coordinator.fest(:,p,K,i) = fest;
        coordinator.eigHest(:,K,i) = eig(coordinator.Hest(:,:,p,K,i));
        % coordinator.err(:,p,K,i)=nan;
      end
    end

    lambdaHist(:,p,K,:) = lambdap;
    if p>=estHorz
      if K>1
        for i=1:subsystems.M
          err = coordinator.hest(1:n/2*(n+1),estHorz,K,i)-coordinator.hest(1:n/2*(n+1),estHorz,1,i);
          coordinator.err(:,p,K,i)=err;
          if norm(err)>1e-4
            % estimate T using hest = T*horig
            T = coordinator.Hest(:,:,estHorz,K,i)*inv(coordinator.Hest(:,:,estHorz,1,i));
            % lambdap(:,i)=inv(T)*lambdap(:,i);
            lambdap(:,i)=-coordinator.Hest(:,:,estHorz,1,i)*theta(:,p,K,i)-inv(T)*coordinator.fest(:,estHorz,K,i);
          end
        end
      end

      lambdaHist(:,p,K,:) = lambdap;
      theta(:,p+1,K,:) = theta(:,p,K,:)+rho*(lambdaHist(:,p,K,:) - mean(lambdap,2));
    end

  end
end
% onlineSecureEstimateHFRand ends here
