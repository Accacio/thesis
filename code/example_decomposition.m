%= Clean variables
close all
clear

paren = @(x, varargin) x(varargin{:}); %
                                       % apply index in created elements
curly = @(x, varargin) x{varargin{:}}; %

%% Options

%= Optimization settings
options = optimset('Display', 'off');
warning off
% options = [];

%= Simulation parameters
Te=.25; %= Sampling
Np=2;   %= Prediction horizon
a=10;  %= for rho=1/(a+b*negot)
b=0.1;  %= for rho=1/(a+b*negot)
simK = 10;    %= Simulation horizon
negotP = 200; %= max # of iteration for each negotiation
err_theta=1e-4; %= err to test theta convergence in each negotiation
rand('seed',2);

%= System
A=[0.8,0.6,0.4];
B=[0.3,0.5,0.6];

%= Initial state
X0(:,1,1) = [3]';
X0(:,1,2) = [2]';
X0(:,1,3) = [1]';

%= Setpoint/References
Wt = [X0(1,1)*1.07;
      X0(1,2)*1.10;
      X0(1,3)*1.05;
     ];
Wt_final = [Wt(1)*1.05;
            Wt(2)*1.05;
            Wt(3)*1.05;
           ];
Wt_change_time =[ simK/2;
                  simK;
                  simK;
                  simK;
                ];
%
%= Global constraint
Umax=4; %= max value

chSetpoint_list = 0; %= Change setpoint?
selfish_list = 0;    %= Selfish behaviour?
secure_list = 0;     %= Secure algorithm?

%= Input bounds
u_min=-inf;
u_max=Umax;

%% Define systems
M=size(A,2);    %= # of systems
%= Define continuos systems using 3R2C
for i=M:-1:1 % make it backward to "preallocate"
    dsys(:,:,1,i)=ss(A(i),B(i),1,0,Te);
    % figure()
    % pzmap(dsys(:,:,1,i))
end

ni=size(dsys.B(:,:,1,1),2); %= # of inputs
ns=size(dsys.A(:,:,1,1),2); %= # of states
n=Np*ni; % constant used everywhere
Gamma(:,:,1)=eye(ni*n);
Gamma(:,:,2)=eye(ni*n);
Gamma(:,:,3)=eye(ni*n);

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
    P(:,:,i)=inv(Gamma(:,:,i)*inv(H(:,:,i))*Gamma(:,:,i).');
end

Ac = kron(ones(M,1),eye(n)); %
                             % Global Constraints
bc = kron(ones(Np,1),Umax);  %

% clear -regexp [^f].*_fun % Delete all functions but f_fun
clear Cwalls* Cair* R* csys


umin(1:M)=u_min;
umax(1:M)=u_max;

%= Selfish behavior profile
% T(:,:,1)=4*eye(n);
T(:,:,1)=20*diag(rand(n,1));
T(:,:,2)=10*eye(n);

%= Time selfish behavior activated
selfish_time= [ simK/2;
                simK;
              ];

rho_fun = @(a,b,negot) 1/(a+b*negot);
% rho = @(a,b,negot) (1/a)/negot;

%% === Control Loop ===
for chSetpoint=chSetpoint_list
for selfish=selfish_list
for secure=secure_list
tic

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
        sHist(:,k,i)=P(:,:,i)*Gamma(:,:,i)*inv(H(:,:,i))*f(:,i);
    end

    modes=2^n;
    PI=repmat(1/modes,1,modes);
    emMaxIter=100;
    maxErr=1e-8;
    %= Negotiation
    %= initialize theta for negotiation
    for i=1:M
        theta(:,1,k,i) = 1*rand(n,1);
    end
    % Projection
    [thetapnew ,~,~,~,~] = quadprog(eye(M*n*ni), -paren(theta(:,1,k,:),':'), ...
                                    [], [], ...
                                    Ac', bc, ...
                                    umin(:,i)*ones(M*ni*n,1), ...  % Lower Bound
                                    umax(:,i)*ones(M*ni*n,1), ...  % Upper Bound
                                    [], options);
    theta(:,1,k,:) = reshape(thetapnew,[n,M]);

    for p=1:negotP

        rho=rho_fun(a,b,p);
        %= Get lambda
        for i=1:M
            % QUADPROG(H,f,A,b,Aeq,beq,LB,UB,X0)
            [u(:,i) ,J(k,i),~,~,l(:,p,k,i)] = quadprog(H(:,:,i), f(:,i), ...
                                                       [], [], ...
                                                       Gamma(:,:,i), theta(:,p,k,i), ...
                                                       umin(:,i)*ones(ni*n,1), ...  % Lower Bound
                                                       umax(:,i)*ones(ni*n,1), ...  % Upper Bound
                                                       [], options);
            % lambda(:,i)=l(:,p,k,i).ineqlin;
            lambda(:,i)=l(:,p,k,i).eqlin;
            if selfish
                if k>selfish_time(i)
                    lambda(:,i)=T(:,:,i)*lambda(:,i);
                end
            end

        end

        lambdaHist(:,p,k,:) = lambda;

        %= Update allocation
        thetap=reshape(theta(:,p,k,:),n,M) + rho*(lambda);
        % thetap=reshape(theta(:,p,k,:),n,M)+rho*(lambda-mean(lambda,2));
        % % Projection
        [thetapnew ,~,~,~,~] = quadprog(eye(M*n*ni), -thetap, ...
                                        [], [], ...
                                        Ac', bc, ...
                                        umin(:,i)*ones(M*ni*n,1), ...  % Lower Bound
                                        umax(:,i)*ones(M*ni*n,1), ...  % Upper Bound
                                        [], options);
        theta(:,p+1,k,:) = reshape(thetapnew,n,1,1,M);
        % theta(:,p+1,k,:) = thetap;


        theta_converged=true;
        for i=1:M
            theta_converged=theta_converged && (norm(theta(:,p+1,k,i)-theta(:,p,k,i),'fro')<=err_theta);
        end
        lastp(k)=p;
        if (theta_converged)
            disp('theta converged');
            break;
        end

    end

    for i=1:M
        u_applied=u(1:ni,i);
        u(1:ni,i)=u_applied;
        % u_applied=0;
        sys=dsys(:,:,1,i);
        xt(:,k+1,i)=sys.A*xt(:,k,i)+sys.B*u_applied;
    end
    uHist(:,k,:) = u(1:ni,:);
end
toc
dataPath='../data/';
%= save data
save(getFileName(dataPath,'example_dmpc','.mat'),'-mat')
end
end
end

for i=1:simK
    figure()
    plot(1:lastp(i),theta(:,1:lastp(i),i,1),'g')
    hold on
    plot(1:lastp(i),theta(:,1:lastp(i),i,2),'b')
    plot(1:lastp(i),theta(:,1:lastp(i),i,3),'r')
    hold off
    legend('\theta_{1_1}','\lambda_{1_2}','\lambda_{2_1}','\lambda_{2_2}','\lambda_{3_1}','\lambda_{3_2}')
    title(['\theta_i k=' num2str(i)])

    figure()
    plot(1:lastp(i),lambdaHist(:,1:lastp(i),i,1),'g')
    hold on
    plot(1:lastp(i),lambdaHist(:,1:lastp(i),i,2),'b')
    plot(1:lastp(i),lambdaHist(:,1:lastp(i),i,3),'r')
    % plot(1:lastp(1),kron(ones(1,lastp(1)),(lambdaHist(:,1,1,1)+lambdaHist(:,1,1,2))/2),'--r')
    plot(1:lastp(i),mean(lambdaHist(:,1:lastp(i),i,:),4),'--k')
    hold off
    legend('\lambda_{1_1}','\lambda_{1_2}','\lambda_{2_1}','\lambda_{2_2}','\lambda_{3_1}','\lambda_{3_2}')
    title(['\lambda_i k=' num2str(i)])
end

figure()
plot(1:simK,uHist(:,1:simK,1))
hold on
plot(1:simK,uHist(:,1:simK,2))
plot(1:simK,uHist(:,1:simK,3))
plot(1:simK,sum(uHist,3))

figure()
hold on
plot(1:simK,xt(:,1:simK,1))
plot(1:simK,kron(ones(1,simK),Wt(1)),'--')
plot(1:simK,xt(:,1:simK,2))
plot(1:simK,kron(ones(1,simK),Wt(2)),'--')
plot(1:simK,xt(:,1:simK,3))
plot(1:simK,kron(ones(1,simK),Wt(3)),'--')
hold off

sympref('FloatingPointOutput',false);
disp(['$a_1=' latex(sym(A(:,1))) '$, $a_2=' latex(sym(A(:,2))) '$, $a_3=' latex(sym(A(:,3))) '$, $b_1=' latex(sym(B(:,1))) '$, $b_2=' latex(sym(B(:,2))) '$, $b_3=' latex(sym(B(:,3))) '$,' ])
% sympref('FloatingPointOutput',true);
disp(['H_1=' latex(sym(H(:,:,1))) ', & \vec{f}_1[k]=' latex(sym(Cmat(:,:,1)'*Qbar(:,:,1)*Mmat(:,:,1))) 'x_i[k]-' latex(sym(Cmat(:,:,1)'*Qbar(:,:,1))) '\vec{W}_i[k], & \bar{\Gamma}_1=I_2,\\\\'])
disp(['H_2=' latex(sym(H(:,:,2))) ', & \vec{f}_2[k]=' latex(sym(Cmat(:,:,2)'*Qbar(:,:,2)*Mmat(:,:,2))) 'x_i[k]-' latex(sym(Cmat(:,:,2)'*Qbar(:,:,2))) '\vec{W}_i[k], & \bar{\Gamma}_2=I_2,\\\\'])
disp(['H_3=' latex(sym(H(:,:,3))) ', & \vec{f}_3[k]=' latex(sym(Cmat(:,:,3)'*Qbar(:,:,3)*Mmat(:,:,3))) 'x_i[k]-' latex(sym(Cmat(:,:,3)'*Qbar(:,:,3))) '\vec{W}_i[k], & \bar{\Gamma}_3=I_2.\\\\'])
