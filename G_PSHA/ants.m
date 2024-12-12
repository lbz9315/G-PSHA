
function [shortest_Length,shortest_Route]=ants(r,A)   
citys=A;

hold on;
 
%% IV. 计算距离矩阵
D = Distance(citys);                                        % 计算距离矩阵   d = distances(G) 返回矩阵 d，其中 d(i,j) 是节点 i 和节点 j 之间的最短路径的长度。
n = size(D, 1);                                             % 城市的个数
 
%% V. 初始化参数
NC_max = 50;                                                
m = n*2;                                                    
alpha = 1;                                                  
beta = 5;                                                   
rho = 0.1;                                                  
Q = 1;
NC = 1;                                                    
 
Eta = 1 ./ D;                                               
Tau = ones(n, n);                                          
Table = zeros(m, n);                                       
rBest = zeros(NC_max, n);                                   
lBest = inf .* ones(NC_max, 1);                             
lAverage = zeros(NC_max, 1);                               
 startpoint=1;    %初始化起点
 endpoint=r;    %初始化终点
 
%% VI. 迭代寻找最佳路径
while NC <= NC_max
    % 第1步：随机产生各个的起点城市
          start = zeros(m,1);
    for i = 1:m
         start(i) = startpoint;    %设置起点
    end
   
    Table(:, 1) = start;                                    % Tabu表的第一列即是所有起点城市
    citys_index = 1: n;                                     % 所有城市索引的一个集合
    % 第2步：逐个路径选择
    for i = 1: m
        % 逐个城市路径选择
        for j = 2: n-1    %迭代到n-1
            tabu = [Table(i, 1: (j - 1)),endpoint];                    % 已经访问的城市集合（称禁忌表），并加入终点endpoint
            allow_index = ~ismember(citys_index, tabu);              % 只访问不在禁忌表里的数据  ismember为判断元素是否属于数组
            Allow = citys_index(allow_index);               % Allow表：存放待访问的城市
            P = Allow;
             
            % 计算从城市j到剩下未访问的城市的转移概率
            for k = 1: size(Allow, 2)                       % 待访问的城市数量
                P(k) = Tau(tabu(end-1), Allow(k))^alpha * Eta(tabu(end-1), Allow(k))^beta;%公式分子，适应度），改为end-1
            end
            P = P / sum(P);                                 % 归一化，这个是转移概率pijk）
             
            % 轮盘赌法选择下一个访问城市（为了增加随机性）
            Pc = cumsum(P);
%             target_index = find(Pc >= rand);   %改，此处总是出错，万一index为0怎么办？？
%             用一个while循环
         while true
              target_index = find(Pc >= rand);
              if  isempty(target_index)
                  continue;
              else
            target = Allow(target_index(1));
            Table(i, j) = target; 
            break;
              end
         end
        end
                   Table(:, n) = endpoint;     %加入终点endpoint
 
    end
     
    % 第3步：计算各个路径距离
    length = zeros(m, 1);
    for i = 1: m
        Route = Table(i, :);
        for j = 1: (n - 1)
            length(i) = length(i) + D(Route(j), Route(j + 1));
        end
   %     length(i) = length(i) + D(Route(n), Route(1));    %不计算最后一段
    end
     
    % 第4步：计算最短路径距离及平均距离
    if NC == 1
        [min_Length, min_index] = min(length);
        lBest(NC) = min_Length;
        lAverage(NC) = mean(length);
        rBest(NC, :) = Table(min_index, :);
    else
        [min_Length, min_index] = min(length);
        lBest(NC) = min(lBest(NC - 1), min_Length);
        lAverage(NC) = mean(length);
        if lBest(NC) == min_Length
            rBest(NC, :) = Table(min_index, :);
        else
            rBest(NC, :) = rBest((NC - 1), :);
        end
    end
   
    Delta_tau = zeros(n, n);
    for i = 1: m         
        for j = 1: (n - 1)
            Delta_tau(Table(i, j), Table(i, j + 1)) = Delta_tau(Table(i, j), Table(i, j + 1)) + Q / length(i);
        end
        % Delta_tau(Table(i, n), Table(i, 1)) = Delta_tau(Table(i, n), Table(i, 1)) + Q / length(i);
    end
    Tau = (1 - rho) .* Tau + Delta_tau;
 
   
    NC = NC + 1;
    Table = zeros(m, n);
end
%% VII. 结果显示
[shortest_Length, shortest_index] = min(lBest);
shortest_Route = rBest(shortest_index, :);

sh=shortest_Length;
 


end
