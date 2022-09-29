function [Phi, pi_new] = update_parameters(Theta, Lambda, Responsibilities)
% UPDATE_PARAMETERS -
    modes=size(Responsibilities,1);
    [c O]=size(Theta);

    Upsilon=kron(ones(c,1),eye(c));
    Delta=kron(eye(O),ones(1,c));
    G=kron(ones(1,O),eye(c));
    Y=kron(G,ones(c,1));
    Omega=sparse([(Upsilon*Theta*Delta).*Y; G]');

    pi_new(:)=sum(Responsibilities,2)/O;
    for i=1:modes
        responsibilities=Responsibilities(i,:);
        resp2=cellfun(@(x) x*eye(c),mat2cell(responsibilities',ones(1,O)),'UniformOutput',0);
        Gamma=sqrt(sparse(blkdiag(resp2{:})));

        Phi(i,:)=-((Gamma*Omega)\(Gamma*Lambda(:)));
    end

end
