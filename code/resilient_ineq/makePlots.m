%% Plots
close all
colors= { '#5A5B9F', '#D94F70', '#009473', '#F0C05A', '#7BC4C4', '#FF6F61'};
linS = {'-k','--k',':k','-.k'};
legJ={'J'};
for i=1:size(umin,2)
  legJ=[legJ,['$J_' num2str(i) '$']];
end
legu={'$u_{max}$','$\Sigma u_i$'};
for i=1:size(umin,2)
  legu=[legu,['$u_' num2str(i) '$']];
end

legErr={};
for i=1:size(umin,2)
  legErr=[legErr,['Sub$_' num2str(i) '$']];
end

set(0, 'DefaultTextInterpreter', 'latex')
set(groot, 'defaultAxesTickLabelInterpreter','latex');
set(groot, 'defaultLegendInterpreter','latex');

figure(1)
clf
title("Temperature ($^oC$)")
hold on
for i=1:M
    stairs(0:simK,xt(1,:,i),linS{i},'Color',colors{i})
end
legend({'I','II','III','IV'})
hold off

figure(2)
clf
title("Input u (Kwh)")
hold on
for i=1:M
    stairs(1:simK,uHist(1,:,i),linS{i},'Color',colors{i})
end
legend({'I','II','III','IV'})
hold off

figure(3)
clf
title("$\theta_i$")
hold on
k=2;
for i=1:M
    for j=1:n
    stairs(1:lastp(k,i),theta(j,1:lastp(k,i),k,i))
    end
end
hold off

figure(4)
clf
title("Lambda")
hold on
k=2;
for i=1:M
    for j=1:n
    stairs(1:lastp(k,i),lambdaHist(j,1:lastp(k,i),k,i))
    end
end
hold off
