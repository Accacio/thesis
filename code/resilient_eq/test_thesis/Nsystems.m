% [[file:../../../docs/org/decomposition_methods/secureDMPC.org::*Define the systems and coordinator][Define the systems and coordinator:2]]
n = 4;
% Define the systems and coordinator:2 ends here

% [[file:../../../docs/org/decomposition_methods/secureDMPC.org::*Define the systems and coordinator][Define the systems and coordinator:3]]
% X0(:,1) = [19 3.2]';
% X0(:,2) = [20. 6.]';

% X0(:,1) = [18.3 3.0]';
% X0(:,2) = [19.6 5.9]';
% X0(:,3) = [18.4 5.3]';

X0(:,1) = [18.3 3.0]';
X0(:,2) = [19.6 5.9]';
X0(:,3) = [18.4 5.3]';
X0(:,4) = [17.4 5.3]';
% Define the systems and coordinator:3 ends here

% [[file:../../../docs/org/decomposition_methods/secureDMPC.org::*Define the systems and coordinator][Define the systems and coordinator:4]]
umin = [0 0 0 0];
umax = [2 2 2 2];

% Umax = 2; % 2 agents
% Umax = 3; % 3 agents
Umax = 4; % 4 agents
% Define the systems and coordinator:4 ends here

% [[file:../../../docs/org/decomposition_methods/secureDMPC.org::*Define the systems and coordinator][Define the systems and coordinator:5]]
linS = {'-k','--k',':k','-.k'};
legJ={'J'};
for i=1:size(umin,2)
  legJ=[legJ,['$J_' num2str(i) '$']];
end
legu={'$u_{max}$','$\Sigma u_i$'};
for i=1:size(umin,2)
  legu=[legu,['$u_' num2str(i) '$']];
end

legErr={};
for i=1:size(umin,2)
  legErr=[legErr,['Sub$_' num2str(i) '$']];
end
% Define the systems and coordinator:5 ends here

% [[file:../../../docs/org/decomposition_methods/secureDMPC.org::*Define the systems and coordinator][Define the systems and coordinator:6]]
% Rf=[5 6];
% Ri=[2.5 2.3];
% Ro=[0.5 1];
% Cres=[5 4];
% Cs=[8 7];
% Rf=[5 6 4];
% Ri=[2.5 2.3 2];
% Ro=[0.5 1 0.8];
% Cres=[5 4 4.5];
% Cs=[8 7 9];

Rf=[5 6 4 5];
Ri=[2.5 2.3 2 2.2];
Ro=[0.5 1 0.8 0.9];
Cres=[5 4 4.5 4.7];
Cs=[8 7 9 6];
% Define the systems and coordinator:6 ends here

% [[file:../../../docs/org/decomposition_methods/secureDMPC.org::*Define the systems and coordinator][Define the systems and coordinator:7]]
M=size(Cs,2);
% Define the systems and coordinator:7 ends here

% [[file:../../../docs/org/decomposition_methods/secureDMPC.org::*Define the systems and coordinator][Define the systems and coordinator:8]]
if chSetpoint
  Wt(:,:,1) = [20*ones(1,floor(simK/2)) 21*ones(1,simK-floor(simK/2))];
else
  Wt(:,:,1) = [20*ones(1,simK)];
end
Wt(:,:,2) = [21*ones(1,simK)];
Wt(:,:,3) = [20*ones(1,simK)];
Wt(:,:,4) = [21*ones(1,simK)];
% Define the systems and coordinator:8 ends here

% [[file:../../../docs/org/decomposition_methods/secureDMPC.org::*Define the systems and coordinator][Define the systems and coordinator:9]]
% 3R-2C model | inputs : u1 = heating; u2 = ext temp
A=@(Cres,Cs,Rf,Ri,Ro) [-1/(Cres*Rf)-1/(Cres*Ri) 1/(Cres*Ri); 1/(Cs*Ri) -1/(Cs*Ro)-1/(Cs*Ri)];
B=@(Cres,Cs,Rf,Ro) [10/Cres 1/(Cres*Rf); 0 1/(Cs*Ro)];
Bp=@(Cres,Cs,Rf,Ro) [10/Cres; 0];
C=[1 0]; D=[0 0];
% Define the systems and coordinator:9 ends here

% [[file:../../../docs/org/decomposition_methods/secureDMPC.org::*Define the systems and coordinator][Define the systems and coordinator:10]]
% Te=0.5; %30min
Te=0.25; %15min
for i=1:M
  psys = c2d(ss(A(Cres(i),Cs(i),Rf(i),Ri(i),Ro(i)),Bp(Cres(i),Cs(i),Rf(i),Ro(i)),C,D(:,1)),Te);
  sys(i).A = psys.A; sys(i).B = psys.B; sys(i).C = psys.C; sys(i).D = psys.D;
  sys(i).umin = umin(:,i);  sys(i).umax = umax(:,i);
  sys(i).xt(:,1) = X0(:,i);  sys(i).Wt = Wt(:,:,i);
end
clear psys
% Define the systems and coordinator:10 ends here

% [[file:../../../docs/org/decomposition_methods/secureDMPC.org::*Define the systems and coordinator][Define the systems and coordinator:11]]
for i=1:M
    Q(:,:,i)=eye(n*size(sys(i).C,1)); % no x no
    R(:,:,i)=eye(n*size(sys(i).B,2)); % nc x nc
end
% Define the systems and coordinator:11 ends here

% [[file:../../../docs/org/decomposition_methods/secureDMPC.org::*Define the systems and coordinator][Define the systems and coordinator:12]]
cheatingProfile=zeros(n*size(sys(1).B,2),n*size(sys(1).B,2),simK,M); % global so cheating functions can see
cheatingProfile(:,:,:,:)=reshape(kron(ones(1,size(sys(1).B,2)*simK*size(sys,2)),eye(n)),[n n size(sys(1).B,2)*simK size(sys,2)]);

tau(:,:,1)=4*eye(n);
tau(:,:,2)=1*eye(n);
tau(:,:,3)=1*eye(n);
tau(:,:,4)=1*eye(n);

cheatingProfile(:,:,floor(end*1/4)+1:end,1)=reshape(kron(ones(1,floor(simK*(3/4))),tau(:,:,1)),[n n size(sys(1).B,2)*floor(simK*(3/4))]);
cheatingProfile(:,:,floor(end*1/2)+1:end,2)=reshape(kron(ones(1,floor(simK*(1/2))),tau(:,:,2)),[n n size(sys(2).B,2)*floor(simK*(1/2))]);
cheatingProfile(:,:,floor(end*3/4)+1:end,3)=reshape(kron(ones(1,floor(simK*(1/4))),tau(:,:,3)),[n n size(sys(3).B,2)*floor(simK*(1/4))]);
cheatingProfile(:,:,floor(end*3/4)+1:end,4)=reshape(kron(ones(1,floor(simK*(1/4))),tau(:,:,4)),[n n size(sys(3).B,2)*floor(simK*(1/4))]);
% Define the systems and coordinator:12 ends here
