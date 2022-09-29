% [[file:../../../docs/org/decomposition_methods/secureDMPC.org::*Define the Control loop][Define the Control loop:6]]
% close all

set(0, 'DefaultTextInterpreter', 'latex')
set(groot, 'defaultAxesTickLabelInterpreter','latex');
set(groot, 'defaultLegendInterpreter','latex');
colors= { '#5A5B9F', '#D94F70', '#009473', '#F0C05A', '#7BC4C4', '#FF6F61'}

figure
for i=1:subsystems.M
    subplot(ceil(subsystems.M/2),2,i);
    stairs(1:K,subsystems.xt(1,1:K,i))
    hold on
    stairs(1:K,subsystems.Wt(1,1:K,i))
    ylim([15 22])
    title(['Output subsystem ' num2str(i) ])
end

figure

subplot(3,1,1);
% Error W-Y
hold on
for i=1:subsystems.M
    stairs(1:K,subsystems.Wt(1,1:K,i)-subsystems.xt(1,1:K,i),linS{i},'Color',colors{i})
end
grid on
title(['Error $w_i(k)-x_i(k)$ '])
set(gcf, 'PaperPosition', [0 0 8 7])
set(gca,'DefaultTextFontSize',18)
set(gca,'FontSize',18)
set(gca,'FontName','cmr18')
set(gcf, 'PaperSize', [8 7])
hx=get(gca,'xlabel');
hy=get(gca,'ylabel');
hz=get(gca,'zlabel');
ht=get(gca,'title');
hl=legend();
set(hx,'FontSize',18)
set(hy,'FontSize',18)
set(hz,'FontSize',18)
set(hl,'FontSize',18)
lgd=legend(legErr,'Location','northeastoutside');
lgd.FontSize = 16;
lgd.FontWeight = 'bold';
% xlabel("Time $k$")
hold off

subplot(3,1,2);

% figure
% Command
hold on
plot(Umax*ones(1,simK),'-r')
plot(sum(reshape(subsystems.uHist(1,end,:,:),[simK subsystems.M]),2)','bo')
for i=1:subsystems.M
    plot(reshape(subsystems.uHist(1,end,:,i),[simK 1]),linS{i},'Color',colors{i})
end
grid on
ylim([0 5])
title(['Command $u_i(k)$'])
set(gcf, 'PaperPosition', [0 0 8 7])
set(gca,'DefaultTextFontSize',18)
set(gca,'FontSize',18)
set(gca,'FontName','cmr18')
set(gcf, 'PaperSize', [8 7])
hx=get(gca,'xlabel');
hy=get(gca,'ylabel');
hz=get(gca,'zlabel');
ht=get(gca,'title');
hl=legend();
set(hx,'FontSize',18)
set(hy,'FontSize',18)
set(hz,'FontSize',18)
set(hl,'FontSize',18)
lgd=legend(legu,'Location','northeastoutside');
lgd.FontSize = 16;
lgd.FontWeight = 'bold';
% xlabel("Time (k)")
hold off

subplot(3,1,3);
% figure
% Norm of estimation error
hold on
for i=1:subsystems.M
    stairs(reshape(vecnorm(coordinator.err(:,end,:,i)),[1 simK]),linS{i},'Color',colors{i})
end
grid on
ylim([-5 40])
title("Norm of error $\| \hat{H_i}-H_{0_i}\|_F$")
set(gcf, 'PaperPosition', [0 0 8 7])
set(gca,'DefaultTextFontSize',18)
set(gca,'FontSize',18)
set(gca,'FontName','cmr18')
set(gcf, 'PaperSize', [8 7])
hx=get(gca,'xlabel');
hy=get(gca,'ylabel');
hz=get(gca,'zlabel');
ht=get(gca,'title');
hl=legend();
set(hx,'FontSize',18)
set(hy,'FontSize',18)
set(hz,'FontSize',18)
set(hl,'FontSize',18)
lgd=legend(legErr,'Location','northeastoutside');
lgd.FontSize = 16;
lgd.FontWeight = 'bold';
% legend(legErr,'Location','northeastoutside','NumColumns',2)
xlabel("Time (k)")
hold off
%save image

path='../../../docs/img/';
filename=[path 'tempErr_Command_HestErr'];
% saveas(gcf,[filename '.pdf'])
% saveas(gcf,[filename '.png'])
ans = filename

% f = figure;
% % Negotiation
% plotnegot(theta,lambdaHist,K);
% b = uicontrol('Parent',f,'Style','slider','Position',[81,54,419,23],...
%               'value',K, 'min',1, 'max',K,'SliderStep',[1/(K-1) 1/(K-1)]);
% bgcolor = f.Color;
% bl1 = uicontrol('Parent',f,'Style','text','Position',[50,54,23,23],...
%                 'String','1','BackgroundColor',bgcolor);
% bl2 = uicontrol('Parent',f,'Style','text','Position',[500,54,23,23],...
%                 'String',num2str(K),'BackgroundColor',bgcolor);
% bl3 = uicontrol('Parent',f,'Style','text','Position',[240,25,100,23],...
%                 'String','K','BackgroundColor',bgcolor);

% b.Callback = @(es,ed) plotnegot(theta,lambdaHist,es.Value) ;

% figure
% EigenValues
% for i=1:subsystems.M
%     subplot(ceil(subsystems.M/2),2,i);
%     stairs(1:K,kron(ones(1,K),eig(subsystems.H(:,:,i)))','r')
%     if isfield(coordinator,'eigHest')
%         hold on
%         stairs(1:K,coordinator.eigHest(:,1:K,i)','b--')
%         hold off
%     end
%     title(['EigenValues subsystem ' num2str(i) ])
% end
% close

% Hest and Epsilon
% if isfield(coordinator,'hest')
%     figure
%     plotWithSlider(@(x,y,p) plot(1:x,y(:,:,p,1)),negotP,coordinator.hest,K)
%     title(['hest 1'])
% end
% if isfield(coordinator,'epsilon')
%     figure
%     plotWithSlider(@(x,y,p) plot(1:x,y(:,:,p,1)),negotP,coordinator.epsilon,K)
%     title(['epsilon 1'])
% end


figure
hold on
plot(sum(reshape(subsystems.J(1,:,:),[simK subsystems.M]),2))
for i=1:subsystems.M
    plot(subsystems.J(1,:,i),linS{i})
end
legend(legJ)
xlabel("Time (K)")
hold off


% close
% Define the Control loop:6 ends here
