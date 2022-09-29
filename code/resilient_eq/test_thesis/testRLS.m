clc
clear all
close all
addpath('../aux')
% options = [];
options = optimset('Display', 'off');
n=3;
simK=4;
negotP=n/2*(n+1)+n;
% n=4;
% simK=2;
% negotP=3;


X0(:,1) = [19 3.2]';
X0(:,2) = [20. 6.]';
umin(:,1)=0;
umax(:,1)=2;
umin(:,2)=0;
umax(:,2)=2;
Wt(:,:,1) = [20*ones(1,simK/2) 20*ones(1,simK/2)];
Wt(:,:,2) = [21*ones(1,simK)];

systems % change systems definitions here
Q(:,:,1)=eye(n*size(ssys(1).C,1)); % no x no
Q(:,:,2)=eye(n*size(ssys(1).C,1)); % no x no
R(:,:,1)=eye(n*size(ssys(1).B,2)); % nc x nc
R(:,:,2)=eye(n*size(ssys(1).B,2)); % nc x nc

subsystems=getSubsystems(ssys,Q,R,n); %no cheating
n=subsystems.n
% generate test data 

theta1=rand(n,negotP);
theta2=rand(n,negotP);
f1=[4 2 6]';
f2=[0.2 1 2]';

lambda1=subsystems.H(:,:,1)*theta1+f1;
lambda2=subsystems.H(:,:,2)*theta2+f2;
theta(:,:,1)=theta1;
theta(:,:,2)=theta2;
lambda(:,:,1)=lambda1;
lambda(:,:,2)=lambda2;
hest=zeros(n/2*(n+1),n,2);
hest=zeros(n/2*(n+1)+n,n,2)
                    

for p = 1:negotP
  for i =1:subsystems.M
    if p==1
      hest_1 = zeros(9,1);
      F_1 = 10*eye(9);
%       F_1 = 10*eye(n/2*(n+1));
    else
      hest_1=hest(:,p-1,i);
      F_1 = F(:,:,p-1,i);
    end
    b=theta(:,p,i);
    upsilon=lambda(:,p,i);
    [hest(:,p,i), F(:,:,p,i), epsilon  ] = estimateHFrls(hest_1, F_1, b, upsilon, 0.2);
  end
end
hest(:,end,:)
subsystems.H(:,:,:)




