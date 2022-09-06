function C = power(A,B)
% Define .^ operator to cells
%   Detailed explanation goes here
C = cellfun(@mpower,A,B,'UniformOutput',false);
end
