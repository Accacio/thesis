% [[file:../../../docs/org/decomposition_methods/secureDMPC.org::applyCheat][applyCheat]]
function lambdas = applyCheat(lambdas,k)
   cheatingProfile = evalin('base','cheatingProfile');
   b(:,:,:)=num2cell(cheatingProfile(:,:,k,:),[ 1 2]);
   % lambdabefore=lambdas
   % lambdasafter=reshape(blkdiag(b{:})*lambdas(:),size(lambdas,1),[])
   lambdas=reshape(blkdiag(b{:})*lambdas(:),size(lambdas,1),[]);
end
% applyCheat ends here
