% [[file:../../../docs/org/decomposition_methods/secureDMPC.org::offlineSecure][offlineSecure]]
function [coordinator, subsystems, theta, lambdaHist] = offlineSecure(coordinator,subsystems,lambdaHist,theta,K,negotP,rho,Ac,bc,proj)
    n = subsystems.n;
    for p=1:negotP

        [ subsystems, lambda ] = getLambdas(subsystems,theta(:,p,K,:),p,K);
        lambdaHist(:,p,K,:) = lambda;
        % pcalcul=4*ceil(subsystems.n/2*(subsystems.n+1));
        pcalcul=ceil(negotP/2);
        if  p==pcalcul
            % K=K
            p=p;
            for i=1:subsystems.M
                theta_unit=theta(:,1:p,K,i);
                lambda_unit=lambdaHist(:,1:p,K,i);
                theta_k=theta_unit(:,2:end);
                theta_k_1=theta_unit(:,1:end-1);
                lambda_k=lambda_unit(:,2:end);
                lambda_k_1=lambda_unit(:,1:end-1);

                F=[kron(eye(n),(theta_k_1-theta_k)');FsymConst(n,n)];
                y = [reshape((lambda_k_1-lambda_k)',1,[])'; zeros((n+1)*n/2-n,1)];

                % F=[kron(eye(n),(theta_k_1-theta_k)')];
                % y=[reshape((lambda_k_1-lambda_k)',1,[])'];
                hflat=-(F'*F)\F'*y;
                coordinator.Hest(:,:,K,i) = round(reshape(hflat,n,n),10);
                H=subsystems.H(:,:,i);
                coordinator.eigHest(:,K,i) = sort(eig(coordinator.Hest(:,:,K,i)));
                eval(['eigH' num2str(i) '=sort(eig(H));']);
                eval(['eigH' num2str(i) 'est=coordinator.eigHest(:,K,i);']);
                % eval(['compVect(eigH' num2str(i) ',eigH' num2str(i) 'est);'])
            end

        elseif K>1 & p>pcalcul
            % K=K
            % p=p
            tauEst = num2cell( mean( round( coordinator.eigHest(:,K,:),2)./round(coordinator.eigHest(:,1,:),2)));
            lambda=lambda/blkdiag(tauEst{:});

            lambdaHist(:,p,K,:) = lambda;
            % coordinator.Hest=
        end

        theta(:,p+1,K,:) = theta(:,p,K,:) + rho*(lambdaHist(:,p,K,:)-mean(lambda,2));

    end

end
% offlineSecure ends here
