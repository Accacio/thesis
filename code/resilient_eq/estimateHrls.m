% [[file:../../../docs/org/decomposition_methods/secureDMPC.org::RLS][RLS]]
function [hest, F, epsilon] = estimateHrls(hest_1, F_1, b, upsilon,forget)
    B=structDataSym(b);
    epsilon=upsilon - B*hest_1;
    F_1;
    F = forget^-1*F_1 - forget^-2*F_1*B'/(eye(size(b,1))+forget^-1*B*F_1*B')*B*F_1;
    hest=hest_1 + F * B' * epsilon;
return
% RLS ends here
