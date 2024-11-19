clear;
clc;
close all; % 关掉显示图形窗口
n1=15;  % n1 为所有寄取点个数  （包括空闲点和需要服务点）***
n=12;    %目标点个数  (寄件点个数（无人机需遍历）)***
n2=4000;
% A3=rand(n1,2)*n2;
% load('A3');
% load('A3(2)');
load('A3');
% load('aaa_A3.mat')


C1=zeros(n1,1);
A1=[A3,C1];
% A=rand(n,2)*n2;

% d=randperm(n1);  % p = randperm(n)   返回一行包含从1到n的整数(随机排列)
% load('d');
% load('d(2)');
load('d');
% load('aaa_d.mat')


A=A1(d(1:n),:); 
A(:,3)=[];
     

% A=randperm(6000,n);   % randperm(m, n)，函数作用：从1-m中随机产生n个不重复的数。  不过是一维数组

C=ones(n,1)*100;
B=[A C];    % B为随机目标点（寄件点）的集合


%在所选的随机50个点中再选出20个点作为打击点

x=5;     %  打击目标点个数(取件点的个数)
% D=datasample(B,x);  %datasample 从矩阵 B 所有元素中随机抽取 一个元素

% d1=randperm(n1);  % p = randperm(n)   返回一行包含从1到n的整数(随机排列)
% load('d1');
% load('d1(2)');
load('d1');
% load('aaa_d1.mat')

D=A1(d1(1:x),:);
D(:,3)=0;     % D为待打击目标点集合（取件点的集合（无人车需遍历））



%% 预处理
%point =A1;   
% 不对，应该只对送取点进行聚类，为了后期统一进行聚类区域划分方便，应该也对取件点加权，只不过因为取件点的重量不考虑，因此权重均为零，而且以送件点优先
% 因为一个点可能同时是送件点和取件点，这时权重按送件点即可
B1=B;
B1(:,3)=0;
B1=[B1;D];
A2=unique(B1,'rows');   %得到关于送件点与取件点的集合
point =A2;  

%对UAV和UGV进行初始定义
Vg=20;   %设定UGV的速度  km/h
%载重
la=10;  %每架UAV 载重10kg

Va=30;  %设置无人机飞行速度 km/h
voy=20000;  % 设置UAV续航为20km

num_UAV = 4;   %UGV搭载4架UAV

%% 用Map 对送件点（UAV服务点）B进行加权赋值   （对取件点加权好像没啥意义）
% J=jiaquan1(A2,B);   % 不应该是对A2进行加权，而只是对B进行加权
% load('J.mat');
% load('J(2).mat');
load('J.mat');
% load('aaa_J.mat');
%% K-均值聚类
%手肘法得到最佳k
k1=shouzhou_k(A1);
% 先验得到k2   利用 ZQ 函数 计算某区域内的总权重
w1=ZQ(A2,J);
k2=ceil(w1/(la*num_UAV));

% k = 6;  % 输入聚类组数
W2=[];  % 用来记录各子区域权重
for k=k2:(k1+k2)
k11=0;
count = 100;  % 定义最大循环次数
[N,~] = size(point);  %N 为100

% （1）s=size(A),
% 
%  当只有一个输出参数时，返回一个行向量，该行向量的第一个元素时矩阵的行数，第二个元素是矩阵的列数。
% （2）[r,c]=size(A),
% 
%          当有两个输出参数时，size函数将矩阵的行数返回到第一个输出变量r，将矩阵的列数返回到第二个输出变量c。

center = point(1:k,:);  % 令前k个点为初始的聚类中心  **************** 需要改
distance_square = zeros(N,k);
while count~=0     %  ~=  表示不等于
    for i = 1:k
        distance_square(:,i) = sum((point - repmat(center(i,:),N,1)).^2,2);
        str = ['Center',num2str(i),'=[];'];
        eval(str);
    end  % 计算每个点到各个聚类中心的距离
    
    for i = 1:N
        minposition = find(distance_square(i,:)==min(distance_square(i,:)));
        str = ['Center',num2str(minposition)];
         eval([str,'=[',str,';point(i,:)];']);
    end  % 建立第一次分类后的分类点集
    
    for i = 1:k
        str = ['Center',num2str(i)];
        eval(['center_New(',num2str(i),',:) = mean(',str,',1);']);
    end  % 计算新的聚类中心
    
    if sym(sum((center_New - center).^2)) == 0
        break
    else
        center = center_New;
    end  % 如果中心未改变则跳出循环
    
    count = count-1;
end
% 得到
for i = 1:k
    I = num2str(i);  % num2str 将数值转换成字符串
%     disp(['第',I,'组聚类的点集为：']);
%     disp(eval(['Center',I]));    % 把聚类点显示出来
    
    b2{i}=eval(['Center',I]);   %取到该次循环第i组的聚类点 
    
    % 对得到的聚类结果进行分析
    % 判断第i组的聚类点是否满足要求  
    w2=ZQ(b2{i},J);
%     m=0;
%     k3=size(b2{i},1);
%     for i=1:k3
%     b22=num2str (b2{i}(i,1));
%     m=m+J(b22);
%     end
    if w2>(la*num_UAV)
        break;
    end
    W2=[W2 w2];
    
    k11=k11+1;
end  
if k11==k  %说明所有子区域均满足要求
    break; %跳出循环
else
    W2=[];
end
 

end


%%

%定义包矩阵，矩阵中每一个元素包含此区域中待打击的目标坐标
%在所选的随机50个点中再选出20个点作为打击点

%调整的重点也在于包矩阵a与b


%存储UGV要服务的点
a=cell(1,k);

for i=1:k
    f=size(b2{i},1);
    for s=1:f
        
        for j=1:x
            tf1 = isequal(b2{i}(s,1),D(j,1));    % 两个完全相同的数值时，tf为1,否则为0
                                               
            if tf1 == 1
                tf2=isequal(b2{i}(s,2),D(j,2));
                if tf1==1&&tf2==1
                a{1,i}=[a{1,i};D(j,:)];    %把每个区域中的目标点分别找出并存在包矩阵a中以待后续使用
                end
                
            end
        end
    end
    
end


b=cell(1,k);  % 存储UAV需要服务的点

for i=1:k
    f=size(b2{i},1);
    for s=1:f
        
        for j=1:n
            tf1 = isequal(b2{i}(s,1),B(j,1));    % 两个完全相同的数值时，tf为1,否则为0
                                               
            if tf1 == 1
                tf2=isequal(b2{i}(s,2),B(j,2));
                if tf1==1&&tf2==1
                b{1,i}=[b{1,i};B(j,:)];    %把每个区域中的目标点分别找出并存在包矩阵b中以待后续使用
                end
                
            end
        end
    end
    
end




%%
%画图聚类
 hold on
% f = figure;
for i = 1:k
    str = ['Center',num2str(i)];
%     plot3(eval([str,'(:,1)']),eval([str,'(:,2)']),eval([str,'(:,3)'])'.','Markersize',15,'color',[rand rand rand]);

     plot3(eval([str,'(:,1)']),eval([str,'(:,2)']),eval([str,'(:,3)']),'*','Markersize',5,'color',[rand rand rand]); % MarkerSize 表示点的大小
     

     eval(['kn = boundary(',str,'(:,1),',str,'(:,2),0.1);',str,'(:,3),']);
     
     hold on
     if isempty(kn)
        eval(['plot(',str,'(:,1),',str,'(:,2));',str,'(:,3),']);
     else
        eval(['plot(',str,'(kn,1),',str,'(kn,2));',str,'(kn,3),']);
     end

end  % 绘图

center(:,3)=0;
hold on

%  plot3(center(:,1),center(:,2),center(:,3),'k+');
 plot3(center(:,1),center(:,2),center(:,3),'kp');
 view(3)
W2
%%
%提取中心点
%  data=center;
%plot (data(:,1),datay(:,2),'-.');
% figure(2)
% plot (center(:,1),center(:,2),"o")







 %% 区域间进行的UGV路径规划  变成蚁群算法，实现UGV的原点出发与回到原点。先得到其大体顺序，即center中心点的一个排序集，放在 p 中
 
[p1,p2]=out_ugv_rootplan(center,k);    %p1 为重新排序后的中心点（包括起点和终点）
                                     %p2 为对应的原来中心点的序列下标


%% 计算并获得各区域无人机的放飞点  放到 F 内存储

center1=p1(1:k+1,:);  %提取p的2到6行，即经过排序后的中心点
b1=cell(1,k);
for i=1:k             %把经过排序后的聚类数组赋给b1
b1{i}=b{p2(1,i)};     
    
end

[Z]=destination (center1,b1,k);  % 得到每一个子区域中距离上一个中心点最近的送取点的位置

%将子区域间UGV的路径进行等分
n=10;  %进行十等分

F=[];
globalMin=[];
T3=[];
disUAV=zeros(100,4);
disUAVs=zeros(1,100);
for i=1:k
points = Equal_parts(center1(i,:),Z(i,:),n);
nn=size(points,1);
for j=1:nn
    T_F=points(j,:);   % T_F   %存储临时起飞点
    xy=[T_F;b1{i};center1(i+1,:)];    %xy即为包含起飞点，送取点，降落点 的一个集合。 把初始点定位xy的第一行所在坐标
    n=size(xy,1);
    
%     min_tour = floor((n-2)/num_UAV);    %设置每个无人机至少搜索过两个目标点（除去起始点和终止点的话，即无人机搜索目标可以为零） 
    min_tour =0;     %最少搜索个数为0，即不要求全部出动
    pop_size = 80;    %设置种群的个数，必须是8的倍数，因为代码中以 8 做为步骤 2 的分组个数       ***********
    num_iter = 100;   %设置迭代总次数， i.e. 5000次
    a11 = meshgrid(1:n);   %用以计算距离矩阵。   a为一个n*n的矩阵，每一行都是1到n
    dmat = reshape(sqrt(sum((xy(a11,:)-xy(a11',:)).^2,2)),n,n); 
    %最新的算法
    [sum_Length,A_shortest_Route,a4,disHistoryUAV,distHistory] = new_yichuan11(xy,dmat,num_UAV,min_tour, pop_size,num_iter,1,1,i,J,la);
    %对照
%     [sum_Length,A_shortest_Route,a4,disHistoryUAV,distHistory] = new_yichuan(xy,dmat,num_UAV,min_tour, pop_size,num_iter,1,1,i,J,la);
    nn1=0;% 记录满足航程的UAV架数
    while nn1<num_UAV
        %先把a2的每一个子矩阵掐头去尾
        xx=size(a4{nn1+1},1);
        a22=a4{nn1+1};
        a22(xx,:)=[];
        a22(1,:)=[];
        w11=ZQ(a22,J)
        if sum_Length(nn1+1)<=voy && w11<la     %%  航程限制与载重限制
            nn1=nn1+1;
        else
            break;
        end
    end
    
    
    %若该子区域的UAV航程均满足要求，则进行画图并显示，随后退出进行到下一个子区域
    if nn1==num_UAV
        F=[F;T_F];
        draw_uav(a4,A_shortest_Route);
        for nn2=1:num_UAV
            disp(['区域',num2str(i),'第',num2str(nn2),'架无人机最短飞行路程为：']);
            disp( num2str(sum_Length(nn2)));
        end
        disp(['区域',num2str(i),'无人机飞行总路程为：']);
        disp( num2str(sum(sum_Length)));
        disp(['区域',num2str(i),'无人机飞行总时间为：']);
        disp( num2str(max(sum_Length)/Va));
        T3=[T3,max(sum_Length)/Va];
        globalMin=[globalMin, sum(sum_Length)];
        disUAV=disUAV+disHistoryUAV;  % 每一架UAV的行驶总距离（每一架UAV在每一个子区域迭代后的历史的和）
        disUAVs=disUAVs+distHistory;  % UAV 的历史总距离迭代数据
         break;
    end

end

end
  disp(['无人机总最短距离:' num2str(sum (globalMin))]);
  uav_dis=sum (globalMin);
%   T3=uav_dis/Va;
  disp(['无人机总最短时间:' num2str(sum(T3))]);

title('改进算法迭代100次UAVs飞行轨迹')
xlabel('区域宽度')
ylabel('区域长度')
zlabel('区域高度')
  
%对所有起飞点进行标记
 plot3(F(:,1),F(:,2),F(:,3),'k^');
 
 
 %%

%区域内进行的UGV路径规划
a1=cell(1,k);
shortest_Length1=[];
for i=1:k             %相应的，把经过排序后的目标点数组赋给a1
a1{i}=a{p2(1,i)};
    
end
center1(1,:)=[];
for i=1:k
    h = size(a1{1,i},1);
    if h~=0
        G=a1{1,i};
        
        H=center1(i,:);
       F1=F(i,:);
     
   disp(['区域',num2str(i),'内部无人车 ']) ;
   [shortest_Length,G_shortest_Route ]= in_ugv(F1,H ,G); 
   disp(['最短距离:' num2str(shortest_Length)]);
   disp(['最短路径:' num2str([G_shortest_Route G_shortest_Route(1)])]);
        
   shortest_Length1=[shortest_Length1,shortest_Length];
    end
end

disp( ['各区域内无人车总距离为 ', num2str(sum(shortest_Length1)) ]) ;

in_ugv_dis=sum(shortest_Length1);

T1=in_ugv_dis/Vg;




%% 绘制UGV 外部轨迹

F2=[F;0,0,0];
p1(k+2,:)=[];
out_ugv_dis1=[];
for i=1:size(p1,1)
   plot3([p1(i,1),F2(i,1)],[p1(i,2),F2(i,2)],[p1(i,3),F2(i,3)],'-ko');
    hold on
end

%计算外部UGV的总距离
for i=1:size(p1,1)
    dis=sqrt((F2(i,1)-p1(i,1))^2 + (F2(i,2)-p1(i,2))^2 );
    disp ([ '第', num2str(i) ,'段距离为： ',num2str(dis) ]);
   
    out_ugv_dis1=[out_ugv_dis1,dis];
end

disp( ['无人车区域间各段总距离为： ',num2str( sum(out_ugv_dis1))]);
out_ugv_dis= sum(out_ugv_dis1);
T2=out_ugv_dis/Vg;





%% 实现无人机的起降，并计算分路程和总路程
%基于遗传算法的旅行商问题
%无人机目标搜索并实现在无人车的起降

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    进度%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% globalMin=[];
% 
%   for i=1:k
%   
% % xy = 10*rand(n,2);   
% xy1=b1{i};    %
% 
% 
% 
% xy=[F(i,:);xy1;center1(i,:)];    %把初始点定位xy的第一行所在坐标
% 
% n=size(xy,1);
% 
% 
% 
% min_tour = 2;    %设置每个无人机至少搜索过两个目标点（除去起始点和终止点的话，即无人机搜索目标可以为零）       
% pop_size = 80;    %设置种群的个数，必须是8的倍数，因为代码中以 8 做为步骤 2 的分组个数       ***********
% num_iter = 5e3;   %设置迭代总次数， i.e. 5000次
% a = meshgrid(1:n);   %用以计算距离矩阵。   a为一个n*n的矩阵，每一行都是1到n
% dmat = reshape(sqrt(sum((xy(a,:)-xy(a',:)).^2,2)),n,n); 
% 
% 
% % [opt_rte,opt_brk,min_dist,globalMin] = test_uav_ugv(xy,dmat,num_UAV,min_tour, pop_size,num_iter,1,1);  %运行代码
%    
% % disp(['区域',num2str(i),'无人机飞行总路程为：']);
% %     disp( num2str(globalMin));
% %      globalMin1=[globalMin1, globalMin];
%  [sum_Length] = yichuan(xy,dmat,num_UAV,min_tour, pop_size,num_iter,1,1,i);
%  disp(['区域',num2str(i),'无人机飞行总路程为：']);
%  disp( num2str(sum(sum_Length)));
%  globalMin=[globalMin, sum(sum_Length)];
%     
%     
%   end
  
%   disp(['无人机总最短距离:' num2str(sum (globalMin))]);
%   uav_dis=sum (globalMin);
%   T3=uav_dis/Va;
%    disp(['无人机总最短时间:' num2str(T3)]);
  
  
  %% 计算总路程
  dis= uav_dis+out_ugv_dis+in_ugv_dis;
  T=max(T1,T3)+T2;
  disp([ '各无人器总航行距离为： ',num2str(dis) ]);
  disp([ '各无人器总航行时间为： ',num2str(T) ]);
  
  hold off;