% [[file:../../../docs/org/decomposition_methods/secureDMPC.org::*symmetric by definition][symmetric by definition:1]]
function y = structDataSym(x)
    s=size(x,1);
    if s==1
        y = x;
        return
    end
    y=[[x';[zeros(s-1,1),x(1)*eye(s-1)]] [zeros(1,s/2*(s -1)); structDataSym(x(2:end))] ];
end
% symmetric by definition:1 ends here
