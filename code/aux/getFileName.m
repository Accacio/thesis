function fullFileName = getFileName(path,baseFilename,ext,varargin)
  filename='';
  n=3;
  if nargin>n
  for i=n+1:nargin
    filename=[filename '_'  inputname(i) '__' num2str(varargin{i-n}) ];
  end
  filename=[filename '_'];
  end
  fullFileName=[path baseFilename filename ext];
end
