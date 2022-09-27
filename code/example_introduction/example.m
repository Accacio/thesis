% [[file:../../../docs/org/decomposition_methods/tricheQuantite.org::*Comportement non-cooperatif d'un agent][Comportement non-cooperatif d'un agent:6]]
lowerb1=-inf;
lowerb2=-inf;
upperb1=inf;
upperb2=inf;
if isunix && ~strcmp(computer,'GLNXA64') || ispc && ~strcmp(computer,'PCWIN64') || ismac && ~strcmp(computer,'MACI64')
  more off
  pkg load optim
end
clearvars -except upperb* lowerb*
close all
options = optimset('Display', 'off');
format long e
addpath("../aux")
Hfun=@(alpha) 2*alpha;
ffun=@(alpha,p) -2*alpha*p;
ctefun=@(alpha,p) alpha*p^2;
Theta1= @(x1) x1;Theta2= @(x2) x2;

alpha = [1 1 3/2 2];
p = [1 3/2 3 5/4];
lb = [lowerb1 lowerb2 lowerb2 lowerb2];
ub = [upperb1 upperb2 upperb2 upperb2];
linS = {'-k','--k',':k','-.k'};

legJ={'J'};
for i=1:size(alpha,2)
  legJ=[legJ,['J' num2str(i)]];
end

x=zeros(size(alpha,2),1);

thetaTotal=1;
theta1=0.2;
theta2=0.3;
theta3=0.3;
thetainit=[theta1 theta2 theta3 thetaTotal-(theta3+theta2+theta1)]';
theta=thetainit;
thetahist=[theta];

H=zeros(1,1,size(alpha,2));
f=zeros(1,1,size(alpha,2));
cte=zeros(1,size(alpha,2));

for i=1:size(alpha,2)
  H(:,:,i)=Hfun(alpha(i));
  f(:,:,i)=ffun(alpha(i),p(i));
  cte(:,i)=ctefun(alpha(i),p(i));
end

Jhist=[];
xhist=[];
tauhist=[];

% taus=[0:0.5:19.5];
% taus=[0:0.5:10];
taus=[ones(1,2),4:1:10,12,13];
% taus=[13:0.000001:13.000001];
tauind=0;
tic
for tau=taus
    theta=thetainit;
    tauind=tauind+1;
    tauhist=[tauhist tau];
    
    J=inf*ones(size(alpha,2),1);
    
    Ji=@(x,H,f) x'*H*x+f'*x;
    eps=1e-10;
    xsol=[];
    lambda=[];
    k=0;
    a=10;b=0;
    rho= @(a,b,k) 1/(a+b*k);
    %rho=@(a,b,k) 0.1*k;
    errX=inf;
    errJ=inf;
    err=@(p) abs(p(end)-p(end-1));
    xnew=zeros(size(H(:,:,1),1),size(alpha,2));
    Jnew=zeros(1,size(alpha,2));
    while errJ>eps & errX>eps & k<=100
      k=k+1;
    
      for i=1:size(alpha,2)
        [xnew(:,i),Jnew(i),~,c(i),lambdanew(i)]=quadprog(H(:,:,i),f(:,:,i),[],[],1,theta(i,end),lb(i),ub(i),x(i,end),options);
      end
    
      x=[x, xnew'];
    
      lambda=[lambda, [tau*lambdanew(1).eqlin lambdanew(2:end).eqlin]'];
      xsol=[xsol,[ c.iterations ]'];
      lambdaMean=mean(lambda(:,end))*ones(size(alpha,2),1);
      theta=theta + rho(a,b,k)*[lambda(:,end)-lambdaMean];
      J=[J,[Jnew+cte]'];
      errJ=err(sum(J,2)');
      errJ=err(J);
      errX=err(sqrt(sum(x.^2)));
      % thetahist=[thetahist theta];
      thetahist(:,k,tauind)=theta;
      lambdahist(:,k,tauind)=lambda(:,end);
    end
    
    xhist=[xhist x(:,end)];
    Jhist=[Jhist J(:,end)];

end
toc
clear W err rhof
dataPath='../../data/example_introduction/';
save(getFileName(dataPath,'example','.mat'),'-mat')
return
%% Figures

f_1=figure('Name','Jhist','NumberTitle','off');
title("Jhist")
f_2=figure('Name','xhist','NumberTitle','off');
figure(f_1)
hold on
plot(tauhist(1:end),sum(Jhist(:,1:end),1),'-r')

for i=1:size(alpha,2)
    plot(tauhist(1:end),Jhist(i,1:end),linS{i})
end

legend(legJ)
ylim([0,20])
hold off

figure(f_2)
hold on
title("xhist")
for i=1:size(alpha,2)
    plot(tauhist(1:end),xhist(i,1:end))
end
ylim([-5,5])
hold off

% Save figure in pdf
% Comportement non-cooperatif d'un agent:6 ends here
