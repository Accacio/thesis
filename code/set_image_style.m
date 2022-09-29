function set_image_style(fontsize)
  set(gca,'FontSize',fontsize);
  set(gca,'FontName','cmr18');
  hx=get(gca,'xlabel');
  hy=get(gca,'ylabel');
  hz=get(gca,'zlabel');
  ht=get(gca,'title');
  set(hx,'FontSize',fontsize);
  set(hy,'FontSize',fontsize);
  set(hz,'FontSize',fontsize);
  set(ht,'FontSize',fontsize);
end
