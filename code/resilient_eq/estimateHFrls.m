% [[file:../../../docs/org/decomposition_methods/secureDMPC.org::*symmetric by definition][symmetric by definition:4]]
function [hest, F, epsilon] = estimateHFrls(hest_1, F_1, b, upsilon,forget)
  B=structDataSym(b);
  B=[ B eye(size(b,1))];
  epsilon=upsilon - B*hest_1;
  F = forget^-1*F_1 - forget^-2*F_1*B'/(eye(size(b,1))+forget^-1*B*F_1*B')*B*F_1;
  hest=hest_1 + F * B' * epsilon;
end
% symmetric by definition:4 ends here
