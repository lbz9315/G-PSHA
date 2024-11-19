% clc;
% clear;
% close;
% load fisheriris
% 
% % data = meas;
% data=rand(100,2)*100;
function [k]=shouzhou_k(A)
data=A;
data(:,3)=[];
[n,p]=size(data);
K=12;

kk=size(data,1);
K=min(K,kk);
D=zeros(K,2);

% K=60;D=zeros(K,2)*12000;
for i=2:K
    
[lable,c,sumd,d]=kmeans(data,i,'dist','sqeuclidean');

sse1 = sum(sumd.^2);
D(i,1) = i;
D(i,2) = sse1;
end

% 自动获取最佳k值
% y=kx+b;
% 斜率k
k1=(D(2,2)-D(K,2))/(D(2,1)-D(K,1));
% b
b=D(2,2)-k1*D(2,1);
Y=[];
C=[];
for i=2:K
  Y=[Y,k1*D(i,1)+b];
  C=[C,abs(Y(i-1)-D(i,2))];
end

[~,k1]=max(C);
k=k1+1

% plot(D(2:end,1),D(2:end,2))
% hold on;
% plot(D(2:end,1),D(2:end,2),'or');
%  hold on;
% 
%  plot([D(2,1),D(K,1)],[D(2,2),D(K,2)])
% xlim([2,8]);
% 
% title('不同K值聚类偏差图') 
% xlabel('分类数(K值)') 
% ylabel('簇内误差平方和') 

