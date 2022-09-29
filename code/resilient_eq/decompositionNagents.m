% [[file:../../../docs/org/decomposition_methods/secureDMPC.org::*Define simulation parameters][Define simulation parameters:1]]
clear
addpath("aux")
paren = @(x, varargin) x(varargin{:}); %
                                       % apply index in created elements
curly = @(x, varargin) x{varargin{:}}; %

options = optimset('Display', 'off');
% rand('seed',5);

% TODO(accacio): change name of these variables
chSetpoint_list = 0; %= Change setpoint?
selfish_list = 0:1;    %= Selfish behaviour?
secure_list = 0:1;     %= Secure algorithm?
% chSetpoint = 0;
% cheating = 0;
% secure = 0;
% Define simulation parameters:1 ends here

% [[file:../../../docs/org/decomposition_methods/secureDMPC.org::*Define simulation parameters][Define simulation parameters:2]]
% simK = 40;
simK = 20;
negotP = 200;
% Define simulation parameters:2 ends here

% [[file:../../../docs/org/decomposition_methods/secureDMPC.org::*Define simulation parameters][Define simulation parameters:3]]
proj=@(x,A,b) x-A/(A'*A)*(A'*x-b);
% Define simulation parameters:3 ends here

for chSetpoint=chSetpoint_list
for selfish=selfish_list
for secure=secure_list
% [[file:../../../docs/org/decomposition_methods/secureDMPC.org::*Define the systems and coordinator][Define the systems and coordinator:1]]
Nsystems
% Define the systems and coordinator:1 ends here

% [[file:../../../docs/org/decomposition_methods/secureDMPC.org::*Define the systems and coordinator][Define the systems and coordinator:13]]
Ac=kron(ones(1,M),eye(n))' ; %
                             % Global Constraints
bc = Umax*ones(1,n)' ;       %
% Define the systems and coordinator:13 ends here

tic
disp(datestr(now,'HH:MM:SS'))

% [[file:../../../docs/org/decomposition_methods/secureDMPC.org::*Define the systems and coordinator][Define the systems and coordinator:14]]
if selfish
  subsystems=getSubsystems(sys,Q,R,n,@applyCheat); % cheating using profile
else
  subsystems=getSubsystems(sys,Q,R,n); %no cheating
end
clear Q R
% Define the systems and coordinator:14 ends here

% [[file:../../../docs/org/decomposition_methods/secureDMPC.org::*Define the systems and coordinator][Define the systems and coordinator:15]]
if secure
%   coordinator=getCoordinator(@onlineSecure); % online secure method using RLS
  % coordinator=getCoordinator(@onlineSecureEstimateCheatMatrix); % online secure method using RLS
  % coordinator=getCoordinator(@onlineSecureEstimateHFCheatMatrix); % online secure method using RLS
  coordinator=getCoordinator(@onlineSecureEstimateHFRand); % online secure method using RLS
  % coordinator=getCoordinator(@onlineSecureEstimateHFRandWaitConvergence); % online secure method using RLS
  % coordinator=getCoordinator(@offlineSecure); % offline secure method
  % coordinator=getCoordinator(@offlineSecureEstimateCheatMatrix); % other offline secure method
else
  % coordinator=getCoordinator(); % no secure method
  coordinator=getCoordinator(@onlineInsecureEstimateHFRand); % online secure method using RLS
end
% Define the systems and coordinator:15 ends here

% [[file:../../../docs/org/decomposition_methods/secureDMPC.org::*Define the Negotiation][Define the Negotiation:1]]
rho = @(a,b,negot) 1/(a+b*negot);
% rho = @(a,b,negot) (1/a)/negot;
a=100;
b=1;

theta=zeros(subsystems.n*subsystems.c,negotP,simK,subsystems.M);
lambdaHist=zeros(subsystems.n*subsystems.c,negotP,simK,subsystems.M);
coordinator.simK=simK;

tic
% Define the Negotiation:1 ends here

% [[file:../../../docs/org/decomposition_methods/secureDMPC.org::*Define the Control loop][Define the Control loop:1]]
for K=1:simK

    for i=1:subsystems.M
        theta(:,1,K,i) = 1*ones(subsystems.n*subsystems.c,1);
    end
% Define the Control loop:1 ends here

% [[file:../../../docs/org/decomposition_methods/secureDMPC.org::*Define the Control loop][Define the Control loop:2]]
    [ coordinator_new, subsystems_new, theta, lambdaHist] = coordinator.negotiate(coordinator,subsystems,lambdaHist,theta,K,negotP,rho(a,b,K),Ac,bc,proj);
    coordinator=coordinator_new;
    subsystems=subsystems_new;
% Define the Control loop:2 ends here

% [[file:../../../docs/org/decomposition_methods/secureDMPC.org::*Define the Control loop][Define the Control loop:3]]
    [ subsystems ] = subsystems.applyControl(subsystems);
% Define the Control loop:3 ends here

% [[file:../../../docs/org/decomposition_methods/secureDMPC.org::*Define the Control loop][Define the Control loop:4]]
end
% Define the Control loop:4 ends here

% [[file:../../../docs/org/decomposition_methods/secureDMPC.org::*Define the Control loop][Define the Control loop:5]]
toc

%% Save data
clear W err rhof
dataPath='../../data/resilient_eq/';
save(getFileName(dataPath,'dmpc4rooms','.mat',chSetpoint,selfish,secure),'-mat')
end
end
end
return
%% Plots
makePlots
% Define the Control loop:5 ends here
