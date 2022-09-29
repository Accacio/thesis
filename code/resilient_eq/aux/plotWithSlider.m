function [] = plotWithSlider(plotfun,x,y,param,fig)
    if nargin==4
        fig=gcf;
    end
    plotfun(x,y,param)
    b = uicontrol('Parent',fig,'Style','slider','Position',[81,54,419,23],...
              'value',param, 'min',1, 'max',param,'SliderStep',[1/(param-1) 1/(param-1)]);
    bgcolor = fig.Color;
    bl1 = uicontrol('Parent',fig,'Style','text','Position',[50,54,23,23],...
                'String','1','BackgroundColor',bgcolor);
    bl2 = uicontrol('Parent',fig,'Style','text','Position',[500,54,23,23],...
                'String',num2str(param),'BackgroundColor',bgcolor);
    bl3 = uicontrol('Parent',fig,'Style','text','Position',[240,25,100,23],...
                'String',inputname(param),'BackgroundColor',bgcolor);
    b.Callback = @(es,ed) plotfun(x,y,ceil(es.Value));
end
