% [[file:../../../docs/org/decomposition_methods/secureDMPC.org::*getLambdasInd][getLambdasInd:1]]
function [subsystems, lambdas ] = getLambdasInd(subsystems,theta,negot,K,i)
    options=evalin('base','options');

    % here we suppose the setpoint won't change in the future
    for i=1:subsystems.M
        newWt(:,:,i)=kron(subsystems.Wt(:,K,i),ones(1,subsystems.n));
        F(:,:,i)=subsystems.getF(subsystems.Cmat(:,:,i), subsystems.Mmat(:,:,i), subsystems.Q(:,:,i), subsystems.xt(:,K,i), newWt(:,:,i)');
        const(:,:,i)= subsystems.getc(subsystems.Mmat(:,:,i), subsystems.Q(:,:,i),subsystems.xt(:,K,i),newWt(:,:,i)');
        % [u1(:,negot,K),J1(1,negot,K),c1,~,lambda1new] = quadprog(H(:,:,1), F(:, :,1), [], [], eye(n*c), theta(:,1)', umin*ones(size(Bd1,2)*n,1), umax*ones(size(Bd1,2)*n,1), [], options);

        %[x,fval,exitflag,output,lambda] = quadprog(H, f, A, b, Aeq, beq, lb, ub, x0, options)

        [u(:,i),J(:,i),~,~,l] = quadprog(subsystems.H(:,:,i), F(:,:,i), [], [], eye(subsystems.n*subsystems.c), theta(:,i)', subsystems.umin(:,i)*ones(subsystems.c*subsystems.n,1), subsystems.umax(:,i)*ones(subsystems.c*subsystems.n,1), [], options);


        lambdas(:,i)=l.eqlin;
    end
    subsystems.u(:,:)=u(:,:);
    subsystems.uHist(:,negot,K,:)=u
    subsystems.J(:,K,:)=J;
    lambdas = subsystems.applyCheat(lambdas,K);
end
% getLambdasInd:1 ends here
