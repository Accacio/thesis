% [[file:../../../docs/org/decomposition_methods/secureDMPC.org::getSubsystems][getSubsystems]]
function subsystems = getSubsystems(ssys,Q,R,n,cheatfunction)
    if (nargin <5)
        cheatfunction = @(x,k) x;
        subsystems.cheatfunctionName = "Nocheating";
    else
        subsystems.cheatfunctionName = func2str(cheatfunction)
    end
% Get mpc params from array of discrete time systems
% and mpc variables Q R and prediction horizon n

paren = @(x, varargin) x(varargin{:}); %
                                       % apply index in created elements
curly = @(x, varargin) x{varargin{:}}; %

% Note implementation of { A, B, C } .* { D, E , F } = { A*D, B*E, C*F}
%  and { A, B, C } .^ { d, e , f } = { A^d, B^e, C^f}

%         _         _
%        | C * A^1  |
% Mmat = | C * A^2  |
%        | C * A^3  |
%        |    .     |
%        |    .     |
%        |    .     |
%        | C * A^n  |
%        |_        _|
Mmat=@(A,C,n) ...
 cell2mat( ...
    (repmat(mat2cell(C,1),1,n) .* ...
     (repmat(mat2cell(A,size(A,2)),1,n).^num2cell(1:n)) ...
    )'  ...
  );
%
%
%
%
% Cmat =  lowertriangle of toeplitz matrix created from
%                       _             _ T
%                      |               |
%                      | C * A^0 * B   |
%                      | C * A^1 * B   |
%                      | C * A^2 * B   |
%                      |       .       |
%                      |       .       |
%                      |       .       |
%                      | C * A^n-1 * B |
%                      |_             _|
Cmat=@(A, B, C, n) ...
    cell2mat( ...
      paren( ...
       (repmat(mat2cell(C, 1), 1, n+1) .* ...
         (horzcat(zeros(size(A)), repmat(mat2cell(A, size(A,2)),1,n).^num2cell(1:n))) .* ...
           repmat(mat2cell(B, size(B,1), size(B,2)), 1, n+1)) ...
           , tril(toeplitz(1:n))+1));


% calculate MPC Variables and functions
getH=@(Cmat,Q,R,n) round(Cmat'*Q*Cmat+R*eye(n),10);
subsystems.getF=@(Cmat,Mmat,Q,xt,Wt) Cmat'*Q*(Mmat*xt-Wt);
subsystems.getc=@(Mmat,Q,Xt,Wt) Xt'*Mmat'*Q*Mmat*Xt-2*Wt'*Q*Mmat*Xt+Wt'*Q*Wt;
subsystems.applyControl = @applyControl;
subsystems.applyCheat = cheatfunction;


subsystems.n = n;
subsystems.M = size(ssys,2); % number of subsystems
subsystems.s= size(ssys(1).A,1);   % number of states  of each subsystem
subsystems.o= size(ssys(1).C,1); % number of outputs of each subsystem
subsystems.c= size(ssys(1).B,2); % number of inputs  of each subsystem
subsystems.Q= Q;
subsystems.R= R;
subsystems.u=[];
subsystems.xt=zeros(subsystems.s,1,subsystems.M);

for i=1:size(ssys,2)
    subsystems.Cmat(:,:,i) = Cmat(ssys(i).A,ssys(i).B,ssys(i).C, n);
    subsystems.Mmat(:,:,i) = Mmat(ssys(i).A,ssys(i).C,n);
    subsystems.H(:,:,i) = getH(subsystems.Cmat(:,:,i), subsystems.Q(:,:,i), subsystems.R(:,:,i), n);
    subsystems.A(:,:,i) = ssys(i).A;
    subsystems.B(:,:,i) = ssys(i).B;
    subsystems.C(:,:,i) = ssys(i).C;
    subsystems.D(:,:,i) = ssys(i).D;
    subsystems.xt(:,1,i) = ssys(i).xt(:,1);
    subsystems.Wt(:,:,i) = ssys(i).Wt(:,:);
    subsystems.umax(:,i) = ssys(i).umax;
    subsystems.umin(:,i) = ssys(i).umin;
end

% TODO rmfield(subsystems,'getF')
% TODO rmfield(subsystems,'getc')
clear getH

% clear unused variables
clear Cmat Mmat
clear Mmat1 Cmat1
clear Mmat2 Cmat2
clear curly paren
end

function [ subsystems ] = applyControl(subsystems)
if isempty(subsystems.u)
    disp("no control defined, vector u empty")
  return
end
t=size(subsystems.xt,2);
for i=1:subsystems.M
  subsystems.xt(:,t+1,i) = subsystems.A(:,:,i)*subsystems.xt(:,t,i)+subsystems.B(:,:,i)*subsystems.u(1,i);
end

end
% getSubsystems ends here
