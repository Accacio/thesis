% [[file:../../../docs/org/decomposition_methods/secureDMPC.org::onlineSecureEstimateCheatMatrix][onlineSecureEstimateCheatMatrix]]
function [coordinator, subsystems, theta, lambdaHist] = onlineSecureEstimateCheatMatrix(coordinator, subsystems, lambdaHist, theta, K, negotP, rho)
    n = subsystems.n;
    mask=triu(true(n));
    forget=1
    for p=1:negotP

        if p==1 & K==1
            coordinator.epsilon = 1*ones(n,p,coordinator.simK,subsystems.M);
            coordinator.F = zeros(n/2*(n+1),n/2*(n+1),p,coordinator.simK,subsystems.M); % must be symmetric
            coordinator.hest = zeros(n/2*(n+1),p,coordinator.simK,subsystems.M);
        end

        thetap=theta(:,p,K,:)
        [ subsystems, lambda ] = getLambdas(subsystems,thetap,p,K);
        lambdaHist(:,p,K,:) = lambda

        if p < 2

        else
            b=thetap-theta(:,p-1,K,:);
            upsilon=lambda-lambdaHist(:,p-1,K,:);
        end

        for i=1:subsystems.M

            disp(['p K i ' num2str(p) ' ' num2str(K) ' ' num2str(i)])
            if p <= 2
                if K < 2
                    if i == 1
                        hest_1 = [4 2 1 3 1 2 ]'
                        % hest_1 = zeros(n/2*(n+1)+n,1)
                        coordinator.hest(:,p,K,i)=hest_1;
                    end
                    if i == 2
                        hest_1 = [4 2 1 3 1 2 ]'
                        % hest_1 = zeros(n/2*(n+1)+n,1)
                        % hest_1 = zeros(6,1)
                        coordinator.hest(:,p,K,i)=hest_1;
                    end
                    F_1 = 10*eye(n/2*(n+1));
                else
                    F_1 = coordinator.F(:,:,end,K-1,i)
                    hest_1 = coordinator.hest(:,end,K-1,i);
                    coordinator.hest(:,p,K,i)=hest_1;
                    coordinator.epsilon(:,p,K,i)=coordinator.epsilon(:,end,K-1,i);
                end

            elseif p > 2
                F_1 = coordinator.F(:,:,p-1,K,i);
                hest_1 = coordinator.hest(:,p-1,K,i);
            end

            if p>1

            [hest, F, epsilon  ] = estimateHrls(hest_1, F_1, b(:,i), upsilon(:,i), forget)

            coordinator.F(:,:,p,K,i) = F;
            coordinator.epsilon(:,p,K,i) = epsilon;
            coordinator.hest(:,p,K,i) = hest;

            H=zeros(n);
            H(mask') = hest;
            coordinator.Hest(:,:,p,K,i) = H'+H-diag(diag(H));
            coordinator.eigHest(:,K,i) = eig(coordinator.Hest(:,:,p,K,i));
            end

        end

        theta(:,p+1,K,:) = theta(:,p,K,:)+rho*(lambdaHist(:,p,K,:) - mean(lambda,2));

    end

end
% onlineSecureEstimateCheatMatrix ends here
