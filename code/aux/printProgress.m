function printProgress(i,itotal,elapsed)
eta=elapsed*itotal/i-elapsed;
if isunix && ~strcmp(computer,'GLNXA64') || ispc && ~strcmp(computer,'PCWIN64') || ismac && ~strcmp(computer,'MACI64')
  Trmsize=terminal_size();
else  
  Trmsize=matlab.desktop.commandwindow.size;
end

disp(['[' repmat('*',[1 ceil((Trmsize(1)-2-1)*i/itotal)]) repmat(' ',[1 ceil((Trmsize(1)-2-1)*(1-i/itotal))]) ']'])
disp(['Elapsed Time: ' num2str(elapsed) 's   ETA: ' num2str(eta) 's it:' num2str(i) ' total:' num2str(itotal) ' = ' num2str(floor(i/itotal*100)) '%' ])

end