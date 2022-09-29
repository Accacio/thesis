close all
clear all

if isunix && ~strcmp(computer,'GLNXA64') || ispc && ~strcmp(computer,'PCWIN64') || ismac && ~strcmp(computer,'MACI64')
  pkg load optim
  %graphics_toolkit gnuplot;
  graphics_toolkit qt;
end

%% Inital X
%% W for each K
%%
umin=0;
umax=2;
umaxSum=2.2;
n=4;
horz=5;
ktotal=4*horz;
% n=horz;
w1beg=20;w1end=20;
w2beg=21;w2end=21;

tau1beg=1;tau1end=1.7;
tau2beg=1;tau2end=1;




X01 = [18.75 3.12]';
X02 = [17.03 5.1614]';

% for umaxsum=2 umax=1.5 [20 20]
X01 = [17.2 3]';
X02 = [17.3 5]';


% % for umaxsum=1.6 [20 20]
% X01 = [14.4 2]';
% X02 = [14.1 4]';

% for umaxsum=2.2 umax=2 [20 20]
X01 = [19 3.2]';
X02 = [20. 6.]';

% Example Romain
% X01 = [10 10]';
% X02 = [10 10]';
% for umaxsum=2
% X01 = [18.77 3.13]';
% X02 = [16.79 5.12]';
for n=[4 8]
for w1end=[20 21]
for tau1end=[1 1.1 1.2 1.3 1.5 1.6 1.7 1.8 1.9]

w1=[w1beg*ones(1,ktotal/2) w1end*ones(1,ktotal/2)];
w2=[w2beg*ones(1,ktotal/2) w2end*ones(1,ktotal/2)];

tau1=[tau1beg*ones(1,ktotal/2) tau1end*ones(1,ktotal/2)];
tau2=[tau2beg*ones(1,ktotal/2) tau2end*ones(1,ktotal/2)];
TAUS=[tau1;tau2];


W=[w1 ;w2];

[J,Y,thetaHist,u1,u2,X,lambda,K,c,H,eigAestHist]=secureDMPC(W,X01,X02,TAUS,n,ktotal,umin,umax,umaxSum);

X011=X01(1);
X012=X01(2);
X021=X02(1);
X022=X02(2);

imgpath="../../../docs/img/";
dataPath='../../../data/matlab/';

%save(getFileName(dataPath,'dmpc2rooms','__withLocalIneq.mat',n,horz,ktotal,X011,X012,X021,X022,w1beg,w1end,w2beg,w2end,tau1beg,tau1end,tau2beg,tau2end,umin,umax,umaxSum),'-mat')
% save(getFileName(dataPath,'dmpc2rooms','.mat',n,horz,ktotal,X011,X012,X021,X022,w1beg,w1end,w2beg,w2end,tau1beg,tau1end,tau2beg,tau2end,umin,umax,umaxSum),'-mat')
end
end
end
% plot(1:K+1,Y(1,:))
% hold on
% stairs(1:K,w1)
% hold off
% ylim([10 25])
% xlim([0 ktotal])

% figure
% % plotyy(1:K+1,Y(2,:),1:K,w2,@plot,@stairs);

% plot(1:K+1,Y(2,:))
% hold on
% stairs(1:K,w2)
% hold off
% ylim([10 25])
% xlim([0 ktotal])

% figure
% plot(1:K,J(1,:))
