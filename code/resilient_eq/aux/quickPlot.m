function h = quickPlot(x,y)
  xLabelName = inputname(1);
  ylabelName =inputname(2);
    
    figure()
    plot(x,y)
    xlabel(xLabelName)
    title(ylabelName)
    
    % ylim(1.1 * [min(y),max(y)])
end
