function plotSystemResponse (A,B,C,u,X0)
ureshaped=reshape(u,size(B,2),[])
[x,y] = simulateSystem(A,B,C,ureshaped,X0);
N=size(ureshaped,2)
plot(1:N+1,y,'-vc')

 
end
