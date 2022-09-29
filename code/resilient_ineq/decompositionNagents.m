%= Clean variables
close all
clear
addpath("../aux")

paren = @(x, varargin) x(varargin{:}); %
                                       % apply index in created elements
curly = @(x, varargin) x{varargin{:}}; %

%% Options

%= Optimization settings
options = optimset('Display', 'off');
warning off
% options = [];
rand('seed',2);

%= Simulation parameters
M=4;    %= # of systems
Te=.25; %= Sampling
Np=4;   %= Prediction horizon

simK = 20;    %= Simulation horizon
negotP = 200; %= max # of iteration for each negotiation
err_theta=1e-3; %= err to test theta convergence in each negotiation

% TODO(accacio): change name of these variables
chSetpoint_list = 0; %= Change setpoint?
selfish_list = 0:1;    %= Selfish behaviour?
secure_list = 0:1;     %= Secure algorithm?

%= Global constraint
Umax=4;
% Umax=2;

%= Input bounds
u_min=0;
% u_min=-inf;
u_max=Umax;
% u_max=inf;

%% Define systems
% TODO(accacio): move to a function

Cair_mean=8;
Cwalls_mean=5;
Roaia_mean=5;
Riwia_mean=2.5;
Rowoa_mean=1.1;

Cwalls = repmat(Cwalls_mean,1,M)+(-.5+rand(1,M));
Cair  = repmat(Cair_mean,1,M)+(-.5+rand(1,M));
Roaia = repmat(Roaia_mean,1,M)+(-.5+rand(1,M));
Riwia = repmat(Riwia_mean,1,M)+(-.5+rand(1,M));
Rowoa = repmat(Rowoa_mean,1,M)+(-.5+rand(1,M));

% values from eq simulation
Cwalls = [8 7 9 6];
Cair = [5 4 4.5 4.7];
Roaia = [5 6 4 5];
Rowoa = [0.5 1.0 0.8 0.9];

%= Define continuos systems using 3R2C
for i=M:-1:1 % make it backward to "preallocate"
    csys(:,:,1,i)=model_3R2C(Roaia(i),Riwia(i),Rowoa(i),Cwalls(i),Cair(i));
end
ni=size(csys.B(:,:,1,1),2); %= # of inputs
ns=size(csys.A(:,:,1,1),2); %= # of states
n=Np*ni; % constant used everywhere
dsys=c2d(csys,Te); %= Discretize systems

%= Output prediction matrices, such $\vec{Y}=\mathcal{M}\vec{x}_k+\mathcal{C}\vec{U}$
% These functions use operator overloading, see https://accacio.gitlab.io/blog/matlab_overload/
Mmat_fun=@(sys,n) ...
         cell2mat( ...
    (repmat(mat2cell(sys.C,1),1,n) .* ...
     (repmat(mat2cell(sys.A,size(sys.A,2)),1,n).^num2cell(1:n)) ...
    )'  ...
                 );
Cmat_fun=@(sys, n) ...
         cell2mat( ...
    paren( ...
    (repmat(mat2cell(sys.C, 1), 1, n+1) .* ...
     (horzcat( ...
    zeros(size(sys.A)), ...
    repmat(mat2cell(sys.A, size(sys.A,2)),1,n).^num2cell(1:n)...
             )) .* ...
     repmat(mat2cell(sys.B, size(sys.B,1), size(sys.B,2)), 1, n+1)) ...
    , tril(toeplitz(1:n))+1));

%= H and f, such $\frac{1}{2}\vec{U}^TH\vec{U}+f^T\vec{U}$
H_fun=@(Cmat,Q,R) round(Cmat'*Q*Cmat+R*eye(size(Q)),10);
f_fun=@(Cmat,Mmat,Q,xt,Wt) Cmat'*Q*(Mmat*xt-Wt);
c_fun=@(Mmat,Q,Xt,Wt) Xt'*Mmat'*Q*Mmat*Xt-2*Wt'*Q*Mmat*Xt+Wt'*Q*Wt;

%= Gains Q and R for $\sum_{j=1}^n \|v\|^2_{Q}+\|u\|^2_{R}$
for i=M:-1:1 % make it backward to "preallocate"
    Qbar(:,:,i)=10*eye(Np*size(dsys(:,:,1,i).C,1)); % no x no
    Rbar(:,:,i)=eye(n); % nc x nc
end

%= Prediction matrices for the systems
for i=M:-1:1
    Mmat(:,:,i)=Mmat_fun(dsys(:,:,1,i),Np);
    Cmat(:,:,i)=Cmat_fun(dsys(:,:,1,i),Np);
    H(:,:,i)=H_fun(Cmat(:,:,i),Qbar(:,:,i),Rbar(:,:,i));
end

Ac = kron(ones(M,1),eye(n)); %
                             % Global Constraints
bc = kron(ones(Np,1),Umax);  %

% clear -regexp [^f].*_fun % Delete all functions but f_fun
clear Cwalls* Cair* R* csys

%= Initial state
% TODO(accacio): move up in code
% TODO(accacio): use random variables?
X0(:,1,1) = [17 3.2]';
X0(:,1,2) = [20 5.3]';
X0(:,1,3) = [15 3.1]';
X0(:,1,4) = [17. 5.7]';

%= Setpoint
Wt = [X0(1,1)*1.50;
      X0(1,2)*1.20;
      X0(1,3)*1.20;
      X0(1,4)*1.20;
     ];
Wt_final = [Wt(1)*1.05;
            Wt(2)*1.05;
            Wt(3)*1.05;
            Wt(4)*1.05;
           ];
Wt_change_time =[ simK/2;
                  simK;
                  simK;
                  simK;
                ];


umin(1:M)=u_min;
umax(1:M)=u_max;

%= Selfish behavior profile
% T(:,:,1)=4*eye(n);
T(:,:,1)=20*diag(rand(n,1));
T(:,:,2)=10*eye(n);
T(:,:,3)=1*eye(n);
T(:,:,4)=1*eye(n);

%= Time selfish behavior activated
selfish_time= [ simK/2;
                simK;
                simK;
                simK;
              ];

rho_fun = @(a,b,negot) 1/(a+b*negot);
% rho = @(a,b,negot) (1/a)/negot;
a=100;
b=100;

%% === Control Loop ===
for chSetpoint=chSetpoint_list
for selfish=selfish_list
for secure=secure_list
tic
disp(datestr(now,'HH:MM:SS'))

u=zeros(n,M);
J=zeros(simK,M);
theta=zeros(n,negotP,simK,M);
lambda=zeros(n,M);
lambdaHist=zeros(n,negotP,simK,M);
uHist=zeros(ni,simK,M);
xt=zeros(ns,simK,M);
xt(:,1,1:M)=X0(:,1,1:M);
lastp=zeros(simK);
norm_err=zeros(simK,M);

for k=1:simK

    rho=rho_fun(a,b,k);
    %= update setpoint?
    % TODO(accacio): add chSetpoint
    if (chSetpoint)
        for i=M:-1:1
            if(k>Wt_change_time(i))
                Wt(:,i)=Wt_final(i);
            end
        end
    end

    %= Get value of f[k]
    for i=M:-1:1
        f(:,i)=f_fun(Cmat(:,:,i),Mmat(:,:,i),Qbar(:,:,i),xt(:,k,i),Wt(i));
        fHist(:,k,i)=f(:,i);
        cHist(k,i)=c_fun(Mmat(:,:,i),Qbar(:,:,i),xt(:,k,i),kron(ones(Np,1),Wt(i)));
    end

    % %= EM
    values=linspace(0,.01,2);

    % see https://accacio.gitlab.io/blog/matlab_combinations/
    [ v{1:n} ]=ndgrid(values);
    em_theta(:,:) =cell2mat(cellfun(@(x) reshape(x,[],1),v,'UniformOutput',0))';

    em_lambda=zeros(n,size(em_theta,2),M);
    em_u=zeros(n,M);
    em_J=zeros(n,M);
    for i=1:M
        for cur_theta=1:size(em_theta,2)
            % QUADPROG(H,f,A,b,Aeq,beq,LB,UB,X0)
            [~,~,~,~,l] = quadprog(H(:,:,i), f(:,i), ...
                                              eye(ni*n), em_theta(:,cur_theta), ...
                                              [], [], ...
                                              umin(:,i)*ones(ni*n,1), ...  % Lower Bound
                                              umax(:,i)*ones(ni*n,1), ...  % Upper Bound
                                              [], options);
            em_lambda(:,cur_theta,i)=l.ineqlin;
            if selfish
                if k>selfish_time(i)
                    em_lambda(:,cur_theta,i)=T(:,:,i)*em_lambda(:,cur_theta,i);
                end
            end
        end
    end

    modes=2^n;
    PI=repmat(1/modes,1,modes);
    emMaxIter=100;
    maxErr=1e-8;
    for i=1:M
        X=em_theta;
        Y=em_lambda(:,:,i);

        Phi_init=20*rand(modes,n^2+n);

        %= Estimate normal behavior
        % [Phi,Responsibilities,~, ~] = emgm_Nestimate (X,Y,[],modes,emMaxIter,maxErr);
        [Phi,Responsibilities,~, ~] = emgm_estimate (X,Y,Phi_init,modes,emMaxIter,maxErr);
        % NOTE(accacio): only if values have zero
        index_of_zero=find(sum(em_theta==zeros(size(em_theta)))==n);
        % index_of_zero=1;
        [~, z_hat_zero]=max(Responsibilities(:,index_of_zero)); %#ok
        zero_params=Phi(z_hat_zero,:);
        H_est(:,:,k,i)=reshape(zero_params(1:n^2),n,n)';
        f_est(:,k,i)=zero_params(n^2+1:end)';

        % display(H_est(:,:,i));
        % display(H(:,:,i));
        % display(f_est(:,i));
        % display(f(:,i));
    end

    %= Negotiation
    %= initialize theta for negotiation
    for i=1:M
        theta(:,1,k,i) = 1*ones(n,1);
    end
    % Projection
    [thetapnew ,~,~,~,~] = quadprog(eye(M*n*ni), -paren(theta(:,1,k,:),':'), ...
                                    Ac', bc, ...
                                    [], [], ...
                                    umin(:,i)*ones(M*ni*n,1), ...  % Lower Bound
                                    umax(:,i)*ones(M*ni*n,1), ...  % Upper Bound
                                    [], options);
    theta(:,1,k,:) = reshape(thetapnew,[n,M]);

    for p=1:negotP

        %= Get lambda
        for i=1:M
            % QUADPROG(H,f,A,b,Aeq,beq,LB,UB,X0)
            [u(:,i) ,J(k,i),~,~,l(:,p,k,i)] = quadprog(H(:,:,i), f(:,i), ...
                                                       eye(ni*n), theta(:,p,k,i), ...
                                                       [], [], ...
                                                       umin(:,i)*ones(ni*n,1), ...  % Lower Bound
                                                       umax(:,i)*ones(ni*n,1), ...  % Upper Bound
                                                       [], options);
            lambda(:,i)=l(:,p,k,i).ineqlin;
            if selfish
                if k>selfish_time(i)
                    lambda(:,i)=T(:,:,i)*lambda(:,i);
                end
            end

        end

        lambdaHist(:,p,k,:) = lambda;
        for i=1:M
            norm_err(k,i)=norm(H_est(:,:,k,i)-H(:,:,i),'fro');
        end
        if secure
            for i=1:M %#ok
                if norm_err(k,i)>1e-4
                    % estimate T using hest = T*horig
                    invT_est=H(:,:,i)/(H_est(:,:,k,i));
                    % disp(invT_est)
                    % disp(inv(T(:,:,i)))
                    lambda(:,i)=-H(:,:,i)*theta(:,p,k,i)-invT_est*f_est(:,k,i);
                end
            end
        end

        %= Update allocation
        thetap=reshape(theta(:,p,k,:),n,M) + rho*(lambda);
        % Projection
        [thetapnew ,~,~,~,~] = quadprog(eye(M*n*ni), -thetap, ...
                                        Ac', bc, ...
                                        [], [], ...
                                        umin(:,i)*ones(M*ni*n,1), ...  % Lower Bound
                                        umax(:,i)*ones(M*ni*n,1), ...  % Upper Bound
                                        [], options);
        theta(:,p+1,k,:) = reshape(thetapnew,n,1,1,M);


        theta_converged=true;
        for i=1:M
            theta_converged=theta_converged && (norm(theta(:,p+1,k,i)-theta(:,p,k,i),'fro')<=err_theta);
        end
        lastp(k)=p;
        if (theta_converged)
            % disp('theta converged');
            break;
        end

    end

    uHist(:,k,:) = u(1:ni,:);
    for i=1:M
        u_applied=u(1:ni,i);
        % u_applied=0;
        sys=dsys(:,:,1,i);
        xt(:,k+1,i)=sys.A*xt(:,k,i)+sys.B*u_applied;
    end
end
toc
dataPath='../../data/resilient_ineq/';
save(getFileName(dataPath,'dmpc4rooms','.mat',chSetpoint,selfish,secure),'-mat')
end
end
end
return
makePlots
