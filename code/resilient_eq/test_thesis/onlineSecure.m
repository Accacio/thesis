% [[file:../../../docs/org/decomposition_methods/secureDMPC.org::onlineSecure][onlineSecure]]
function [coordinator, subsystems, theta, lambdaHist] = onlineSecure(coordinator, subsystems, lambdaHist, theta, K, negotP, rho,Ac,bc,proj)
    n = subsystems.n;
    mask = triu(true(n));

    for p=1:negotP

        if p==1 & K==1
            coordinator.epsilon = 100*ones(n,p,coordinator.simK,subsystems.M);
            coordinator.F = zeros(n/2*(n+1),n/2*(n+1),p,coordinator.simK,subsystems.M); % must be symmetric
            coordinator.hest = zeros(n/2*(n+1),p,coordinator.simK,subsystems.M);
        end

        disp(['p K ' num2str(p) ' ' num2str(K)])

        thetap=theta(:,p,K,:);
        [ subsystems, lambda ] = getLambdas(subsystems,thetap,p,K);
        lambdaHist(:,p,K,:) = lambda;

        if p == 1
            b=thetap;
            upsilon=lambda;
        else
            b=thetap-theta(:,p-1,K,:);
            upsilon=lambda-lambdaHist(:,p-1,K,:);
        end

        b
        upsilon
        for i=1:subsystems.M
            B=kron(eye(n),b(:,i)')
            H=zeros(n);
            if p==1
                if K<2
                    % symmetry constraint
                    B=[B;FsymConst(n,n)]

                    upsilon
                    F_1 = eye(n/2*(n+1))
                    hest_1 = zeros(n/2*(n+1),1);
                else
                    F_1 = coordinator.F(:,:,end,K-1,i)
                    hest_1 = coordinator.hest(:,end,K-1,i);
                end

            else
                F_1 = coordinator.F(:,:,p-1,K,i);
                hest_1 = coordinator.hest(:,p-1,K,i);
            end

            % if p>1
            epsilon=upsilon(:,i) - B*hest_1;
            coordinator.epsilon(:,p,K,i) = epsilon;
            F = F_1 - F_1*B'/(eye(n)+B*F_1*B')*B*F_1
            coordinator.F(:,:,p,K,i) = F;

            hest=hest_1 + F * B' * epsilon
            coordinator.hest(:,p,K,i) = hest;
            coordinator.hest(:,p,K,i) = hest;

            H(mask') = hest;
            coordinator.Hest(:,:,p,K,i) = H'+H-diag(diag(H));
            coordinator.eigHest(:,p,K,i) = eig(coordinator.Hest(:,:,p,K,i));
            % end

        end

        theta(:,p+1,K,:) = theta(:,p,K,:)+rho*(lambdaHist(:,p,K,:) - mean(lambda,2));

    end

end
% onlineSecure ends here
