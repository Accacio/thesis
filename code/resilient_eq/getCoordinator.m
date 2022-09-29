% [[file:../../../docs/org/decomposition_methods/secureDMPC.org::*define Coordinateur][define Coordinateur:1]]
function coordinator = getCoordinator(mynegot)
    if (nargin < 1)
        coordinator.negotiate = @negotiate;
        coordinator.negotiateName = "NoSecure";
    else
        coordinator.negotiate = mynegot;
        coordinator.negotiateName = func2str(mynegot);
    end
end

function [coordinator, subsystems, theta, lambdaHist] = negotiate(coordinator, subsystems, lambdaHist, theta, K, negotP, rho, Ac, bc, proj)

    for p=1:negotP
        [ subsystems, lambda ] = getLambdas(subsystems, theta(:,p,K,:), p, K);

        lambdaHist(:,p,K,:) = lambda;

        theta(:,p+1,K,:) = theta(:,p,K,:) + rho*(lambdaHist(:,p,K,:)-mean(lambda,2));

        % TODO use projection
        % tproj = proj(thetap(:),Ac,bc);
        % theta(:,p+1,K,:) = theta(:,p,K,:) + rho*(lambdaHist(:,p,K,:)-mean(lambda,2));

    end

end
% define Coordinateur:1 ends here
