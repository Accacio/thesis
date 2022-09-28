function C = times(A,B)
% Define .* operator to cells
%   Detailed explanation goes here
C = cellfun(@mtimes,A,B,'UniformOutput',false);
end
