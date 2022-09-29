function compVect(varargin)
  names=[''];
  displ=[''];
  for i=1:nargin
    length(inputname(i));
    names=[names, ' | ' repmat([' '],1,floor((13-length(inputname(i)))/2))  inputname(i) repmat([' '],1,floor((13-length(inputname(i)))/2))];
    displ=[displ, ' | %8.10f'];
  end
  names=[names, ' | '];
  displ=[displ, ' | '];
  fprintf([names '\n'])
  
  fprintf([displ '\n'],[varargin{:}]')
%  fprintf(['|' getVarName(varargin)\n'])
  end