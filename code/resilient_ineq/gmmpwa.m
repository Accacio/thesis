clear; close all ;
subplot(1,2,1)
x1 = linspace(0,5,100);
x2 = linspace(-5,10,100);
[X1 ,X2 ] = meshgrid(x1,x2);
Z=(1000/sqrt(2*pi).*exp(-((X2-0*X1-0).^2/2)));;

surf(X1,X2,Z,'LineStyle','none')
colormap(gca,bone)
set_image_style(18);
view(0,90)

subplot(1,2,2)
x1 = linspace(0,5,100);
x2 = linspace(-5,10,100);
[X1 ,X2 ] = meshgrid(x1,x2);
Z=(1000/sqrt(2*pi).*exp(-((X2-2*X1+4).^2/2)));;
surf(X1,X2,Z,'LineStyle','none')
colormap(gca,bone)
sgtitle('Gaussian distributions used in mixture','FontSize',18,'interpreter','latex')
view(0,90)
set_image_style(18);
p
% print_image_pdf('../img/resilientpwagaussian',[8 3.5])
