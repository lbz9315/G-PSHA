function DrawPath(Chrom, X)
%% 画路线图函数
% 输入：
% Chrom：待画路线
% X：各目标的坐标位置
% R = [Chrom(1, :) Chrom(1, 1)];  % 一个随机解（个体）    Chrom(1, 1)表示又把起点加上了，从而形成闭路
R =Chrom(1, :);
% figure;
hold on
plot3(X(:, 1), X(:, 2),X(:,3) ,'o', 'color', [0.5, 0.5, 0.5])
plot3(X(Chrom(1, 1), 1), X(Chrom(1, 1), 2), X(Chrom(1, 1), 3), 'rv', 'MarkerSize', 20)      %标三角符号以示开始点

for i = 1: size(X, 1)
    text(X(i, 1) + 0.05, X(i, 2) + 0.05, num2str(i), 'color', [1, 0, 0]);     %给中心点标号
end
A = X(R, :);            %A是将之前的坐标顺序用R打乱后重新存入A中
row = size(A, 1);          %row为坐标数+1

%改为三维的，为z轴赋值
arrowz=zeros(2,1);

for i = 2: row
  
%     [arrowx, arrowy] = dsxy2figxy( gca, A(i - 1: i, 1), A(i - 1: i, 2));        %dsxy2figxy坐标转换函数，记录两个点
    %  annotation('textarrow', arrowx, arrowy, 'HeadWidth', 8, 'color', [0, 0, 1]);   %将这两个点用带箭头的线段连接起来 
  plot3( A(i - 1: i, 1), A(i - 1: i, 2),arrowz,'-k');
end
view(3)
% hold off
xlabel('X(m)')
ylabel('Y(m)')
zlabel('Z(m)')
% title('轨迹图')
box on  