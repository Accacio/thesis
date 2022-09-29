function [Phi,Responsibilities,pi_new,Sigma] = emgm_estimate(X,Y,phi_init,modes,emMaxIter,maxErr)
% EMGM_NESTIMATE - ESTIMATE N DIMENSIONAL

    Pi=repmat(1/modes,1,modes);
    OldclusterSize=zeros(1,modes);
    [n, ~]=size(X);

    if(isempty(phi_init))
        % TODO(accacio): do some multidimensional magic
        % indexpts=randi([1 size(x,2)-1],1,modes);
        % dx=max(x(:,2:end)-x(:,1:end-1),[],2);
        % dx=num2cell(dx,2);

        % y_square=reshape(y,sqrt(size(y,2)),sqrt(size(y,2)));
        % [V{1:2}]=gradient(y_square,dx{:});
        % grad=cell2mat(cellfun(@(x) reshape(x,[],1),V,'UniformOutput',0))';
        % C=grad(:,indexpts);
        % x_indexed=x(:,indexpts);
        % y_indexed=y(:,indexpts);
        % d=y_indexed-sum(C.*x_indexed);
        % C=reshape(C,2,1,modes);
        % d=reshape(d,1,1,modes);
        % C=5*rand(size(x,1),1,modes);
        % d=5*rand(1,1,modes);
        Phi=20*rand(modes,n^2+n);

    else
        Phi=phi_init;
    end

    OldPhi=zeros(size(Phi));
    % eps=100;
    eps=10000;
    Sigma(:,:,1:modes)=repmat(eps*eye(n),1,1,modes);

    for emInd=1:emMaxIter

        Responsibilities=calculate_responsibilities(X,Y,Phi,Sigma,Pi);

        [Phi, pi_new] = update_parameters(X, Y, Responsibilities);

        [~,z_hat]=max(Responsibilities,[],1);
        for i=1:modes
            z_i=find(z_hat==i);
            clusterSize(i)=size(z_i,2);
            if OldclusterSize(i)==clusterSize(i)
                % Sigma(:,:,i)=Sigma(:,:,i)*.9;
                Sigma(:,:,i)=Sigma(:,:,i)*.01;
            else
                Sigma(:,:,i)=Sigma(:,:,i)*1.;
            end
        end

        % Phi
        if sum(sum(sum(abs(Sigma)<maxErr,3)==modes*ones(n),2))==n^2
            break;
        end
        if abs(OldPhi-Phi)<maxErr
            break;
        end
        OldclusterSize=clusterSize;
        OldPhi=Phi;
    end
end
