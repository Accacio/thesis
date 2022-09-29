% [[file:../../../docs/org/decomposition_methods/secureDMPC.org::systems][systems]]
%define system params
% room 1    % room 2
Rf1=5;      Rf2=6;
Ri1=2.5;    Ri2=2.3;
Ro1=0.5;    Ro2=1;
Cres1=5;    Cres2=4;
Cs1=8;      Cs2=7;

%define how to calculate using 2C3R
% inputs : u1 = heating; u2 = ext temp
A=@(Cres,Cs,Rf,Ri,Ro) [-1/(Cres*Rf)-1/(Cres*Ri) 1/(Cres*Ri) ;
    1/(Cs*Ri) -1/(Cs*Ro)-1/(Cs*Ri)];
B=@(Cres,Cs,Rf,Ro) [10/Cres 1/(Cres*Rf);
    0 1/(Cs*Ro)];
C=[1 0];
D=[0 0];

% room 1                         % room 2
A1=A(Cres1,Cs1,Rf1,Ri1,Ro1);     A2=A(Cres2,Cs2,Rf2,Ri2,Ro2);
B1=B(Cres1,Cs1,Rf1,Ro1);         B2=B(Cres2,Cs2,Rf2,Ro2);
C1=C;                            C2=C;
D1=D;                            D2=D;
sys1=ss(A1,B1,C1,D1);            sys2=ss(A2,B2,C2,D2);

Te=0.5; % sampling time = 30 min

% room 1              % room 2
sys1d=c2d(sys1,Te);   sys2d=c2d(sys2,Te);
Ad1=sys1d.a;          Ad2=sys2d.a;
Bd1=sys1d.b(:,1);     Bd2=sys2d.b(:,1);
Cd1=sys1d.c;          Cd2=sys2d.c;
clear sys1d sys2d

sys1d.A = Ad1;        sys2d.A = Ad2;
sys1d.B = Bd1;        sys2d.B = Bd2;
sys1d.C = Cd1;        sys2d.C = Cd2;
sys1d.D = [];         sys2d.D = [];

sys1d.umin = umin(:,1);
sys1d.umax = umax(:,1);
sys1d.xt(:,1,1) = X0(:,1);
sys1d.Wt(:,:) = Wt(:,:,1);

sys2d.umin(:,1) = umin(:,2);
sys2d.umax(:,1) = umax(:,2);
sys2d.xt(:,1,2) = X0(:,2);
sys2d.Wt(:,:) = Wt(:,:,2);


ssys(1)=sys1d;
ssys(2)=sys2d;

clear Cres1 Cs1 Ri1 Ro1 Rf1
clear Cres2 Cs2 Ri2 Ro2 Rf2
clear A B C D
clear A1 B1 C1 D1
clear A2 B2 C2 D2
clear Te
clear Ad1 Bd1 Cd1
clear Ad2 Bd2 Cd2
clear sys1 sys1d
clear sys2 sys2d
% systems ends here
