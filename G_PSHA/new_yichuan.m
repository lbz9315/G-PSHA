
function [sum_Length,shortest_Route,a4,disHistoryUAV,distHistory] = new_yichuan(xy,dmat,nSalesmen,minTour,popSize,numIter,showProg,showResult,i,M,la)

hold on

% Verify Inputs  验证输入是否可行，验证原理为城市个数 N 是否和 距离矩阵的 size相等
  [N,dims] = size(xy);     
  [nr,nc] = size(dmat);
  
% （1）s=size(A),
% 
%          当只有一个输出参数时，返回一个行向量，该行向量的第一个元素时矩阵的行数，第二个元素是矩阵的列数。

% （2）[r,c]=size(A),
% 
%          当有两个输出参数时，size函数将矩阵的行数返回到第一个输出变量r，将矩阵的列数返回到第二个输出变量c。
% 
% （3）size(A,n)如果在size函数的输入参数中再添加一项n，并用1或2为n赋值，则 size将返回矩阵的行数或列数。其中r=size(A,1)该语句返回的时矩阵A的行数， c=size(A,2) 该语句返回的时矩阵A的列数。

D=sqrt((xy(1,1) - xy(N,1))^2 + (xy(1,2) - xy(N,2))^2);   %算一下起飞点降落点间的直线距离，来矫正未起飞飞机的飞行距离
if N ~= nr || N ~= nc
    error('Invalid XY or DMAT inputs!')
end
n = N-2;  %去掉了起始点和结束点************************************************ 还是得改过来
 
% % Sanity Checks    验证输入：可以不看
% nSalesmen = max(1,min(n,round(real(nSalesmen(1)))));  
% %验证输入的旅行商个数是不是大于1，并且是整数，否则帮你四舍五入改了
%  minTour = max(1,min(floor(n/nSalesmen),round(real(minTour(1)))));  %*****
% %验证输入的minTour是不是大于1，并且是整数，否则帮你四舍五入改了       round函数负责进行四舍五入     
% popSize = max(8,8*ceil(popSize(1)/8));
% %验证输入的个体数是否为8的整数（因为后面的分组8个为一组），否则帮你用ceil函数改了
% numIter = max(1,round(real(numIter(1))));
% %验证输入的迭代次数是否大于1，否则帮改了
% showProg = logical(showProg(1));      %logical函数是把数值变成逻辑值，logical(x)将把x中的非0的值 变成1，把所有的数值0值变成逻辑0
% %验证是否为1或0，下同
% showResult = logical(showResult(1));
 
 
% Initializations for Route Break Point Selection    路由断点选择的初始化  
nBreaks = nSalesmen-1;    %设置中断点个数。         *****************中断个数为什么比旅行商少1 ？？？？？？？？？
dof = n - minTour*nSalesmen;          % degrees of freedom    自由度    =  中间城市数 - 每个人最少经历城市数*人数
addto = ones(1,dof+1);  
for k = 2:nBreaks
    addto = cumsum(addto);      % cumsum 为累积和  A = 1:5;
                                                 % B = cumsum(A)
                                                 % B =    1     3     6    10    15

end
cumProb = cumsum(addto)/sum(addto);   % sum( a , 1 )                  % 意思即为对矩阵 a 的列求和
                                      % sum( a , 2 ) = sum( a )       % 意思即为对矩阵 a 的行求和
                                      % cumProb计算累积概率。                  
 
% Initialize the Populations
popRoute = zeros(popSize,n);         %population of routes，popRoute 为所有个体的路径基因型     种群数量*中间城市个数 的矩阵
popBreak = zeros(popSize,nBreaks);   %population of breaks     所有个体的中断点                 种群数量*中断点个数  的矩阵
for k = 1:popSize                    %popSize 种群个数
    popRoute(k,:) = randperm(n)+1;   %随机产生所有个体的路径基因型，下同。           %p = randperm(n)    返回一行包含从1到n的行向量
    %为什么＋1？别忘了上面有一句 n = N-2，所以应该是 2到34的随机序列才对。             %p = randperm(n,k)  返回一行从1到n的整数中的k个元素的行向量，而且这k个数也是不相同的
    %随机给路径矩阵赋值（按行）
    %*******************************************************************
    popBreak(k,:) = rand_breaks();   %rand_breaks()为产生中断点的代码，在下面呢。   产生一系列中断基因（中断基因初始化）*****************************
end
 
 
%画图时，将每一个旅行商们走的路用不用颜色标出来。
% pclr = ~get(0,'DefaultAxesColor');
% clr = [1 0 0; 0 0 1; 0.67 0 1; 0 1 0; 1 0.5 0];
% if nSalesmen > 5
%     clr = hsv(nSalesmen);
% end
 
% Run the GA
% globalMin = Inf; %初始化全局最小值。设为无穷大，等着被换的家伙。
totalDist = zeros(1,popSize);  %初始化总距离，是一个行向量，每一个个体对一应一个总距离
% totalDist =[];
distHistory = zeros(1,numIter);   %历史距离，用于比较最好的距离，每一次迭代，都产生一最好距离作为历史距离存起来。    numIter为迭代次数
% distHistory = zeros(1,30);%*******************************************************************************
tmpPopRoute = zeros(8,n);             
%暂时变量，用完就丢。用于产生新个体的，（路径的基因型）
tmpPopBreak = zeros(8,nBreaks);
%同上，用于产生新的中断点的基因型
newPopRoute = zeros(popSize,n);
%新生代的路径基因型  
newPopBreak = zeros(popSize,nBreaks);
%新生代的断点基因型


% if showProg
%     pfig = figure('Name','MTSPOF_GA | Current Best Solution','Numbertitle','off');       %创建一个图窗口的名称
% end
%画图：初始点

a2=cell(1,nSalesmen);    %创建一个包矩阵，将每一个区域内每一架无人机经历的点的集合存起来。用遗传算法来给无人机路径分组，再用蚂蚁算法给每个组的点进行路径规划

% for iter = 1:numIter         % 如果要更改，这里应该是重点，由于粒子群算法的加入，可以大大减少迭代的次数*****************************
    % Evaluate Members of the Population
BG=Inf;
UAVsum=Inf;  %初始化UAV总飞行长度，为最大值
disHistoryUAV=zeros(numIter,nSalesmen);
for iter = 1:100    %  迭代次数改为50 *********************************
    for p = 1:popSize
%         d = 0;
        %**************************可能恰巧每一对基因与其中断基因所构成的都不能满足约束，与其
        pRoute = popRoute(p,:);  
        %将相应的个体的路径基因型取出
        pBreak = popBreak(p,:);
        %将相应的个体的中断点基因型取出
%         for j=1:nSalesmen
%             for jj=1:
%         end
        
  
        rng = [[1 pBreak+1];[pBreak n]]';
        %计算每个旅行商的距离之用
        %下面的迭代用于计算每个个体的对应的所有旅行商的总距离

          %改
          %**********************************************************************************
          % 先取到每一段的路径，检查是否满足约束
%         for s = 1:nSalesmen
%            if s==1
%               a2{1}=[ xy(rng(1,1):rng(1,2),:);xy(rng(nSalesmen,2),:)]; 
%            elseif s~=1&&s~=nSalesmen
%                a2{s}=[ xy(rng(1,1),:);xy(rng(s,1):rng(s,2),:);xy(rng(nSalesmen,2),:)];
%            else
%                a2{s}=[ xy(rng(1,1),:);xy(rng(s,1):rng(s,2),:)];
%            end
%       
%         end
        for s = 1:nSalesmen
            a2{s}=[xy(pRoute(rng(s,1):rng(s,2)),:)];  %创建一个包矩阵，将每一个区域内每一架无人机经历的点的集合存起来。用遗传算法来给无人机路径分组
            a3{s}=pRoute(rng(s,1):rng(s,2));
        end
        % 这里属于标记待调整部分    
        nn=0;
        for s = 1:nSalesmen
          if ZQ(a2{s},M)<=la 
              ZQ(a2{s},M)
              nn=nn+1;
          else
            a2={};  % 如果该路径基因不满足约束，拟把该行换成全零行（或者把该行第一项变为0作为标志即可），后期找到最佳符合路径和对应中断时，用最佳路径来代替此全零行同时中断点也进行更换，以此实现种群的迭代优化
            a3={};
%             popRoute(p,1)=0;  % 做标记
            totalDist(p)=Inf;  % 将不符合约束的基因所得到路径长度都记为无穷大
            break;
          end
        end
   
        if nn~=nSalesmen
            continue;
        else
            %  要改应该是改这里，还要不要让他进行  5000*80  次的迭代？***************************************************        
%             sum_Length=[];
            for qq=1:nSalesmen
               a2{qq}=[xy(1,:);a2{qq};xy(N,:)]; 
            end
            length=[];
%             shortest_Route={};
            
            pre_popRoute=[]; % 记录新序列以替换旧序列
            for s = 1:nSalesmen
              [r,~]=size(a2{s});
              [shortest_Length,shortest_Route1 ]= ants(r,a2{s});      %*************这里返回的是相对位置序列
        %     disp(['区域',num2str(i),'第',num2str(s),'架无人机最短飞行路程为：']);
        %     disp( num2str(shortest_Length));
        %矫正不起飞无人机的飞行距离
              if shortest_Length==D  
                  shortest_Length=0;   %矫正不起飞无人机的飞行距离
                  shortest_Route1=[1,1];
              end
              length=[length, shortest_Length];
              sRoute{s}=shortest_Route1;
              
            
              %因为返回的是相对位置，因此需要根据相对位置调整原数组后再掐头去尾，然后再加到pre_popRoute里
              
              si=size(shortest_Route1,2);
              shortest_Route1(si)=[];
              shortest_Route1(1)=[];
              a3{s}=a3{s}(:,shortest_Route1-1);   %根据相对位置调整原路径数组
              pre_popRoute=[pre_popRoute,a3{s}];
            end
            popRoute(p,:)=pre_popRoute;  % 替换原序列
            totalDist(p)=sum(length); %记录每一个种群个体的总路径长
            HisdisUAV(p,:)=length;   %记录每一次迭代各UAV的路径长
            if(sum(length)<UAVsum)
               UAVsum=sum(length);
               sum_Length=length;     %存储 80次迭代后的各UAV最短距离
               shortest_Route=sRoute; %存储 80次迭代后 在该中断序列下 的各UAV最优路径 （元胞数组） 
               a4=a2;                 %存储各UAV最优的序列
                
               
               %获得 80次迭代最优的路径和中断序列，用来对不满足约束的个体进行替换
               Best_R=popRoute(p,:);  %取到最优路径排列  
               Best_B=popBreak(p,:);   %取到对应的中断序列

            end  
            a2={};
            a3={};
        end
    end
    [min_dist,index] = min(totalDist);     %[M,I] = min(A)  创建一个矩阵 A 并计算每列中的最小元素，以及这些元素在 A 中显示的行索引。前者组成行向量给M，后者组成行向量给I。
    distHistory(iter) = min_dist;
    if min_dist==Inf
        disHistoryUAV(iter,:)=zeros(1,4);
        sum_Length=Inf;
        shortest_Route={[1,2],[1,2],[1,2],[1,2]};
        a4={[1,2],[1,2],[1,2],[1,2]};
        continue;
    elseif min_dist<BG
    BG=min_dist;
    
    disHistoryUAV(iter,:)=HisdisUAV(index,:);
    else
        disHistoryUAV(iter,:)=disHistoryUAV(iter-1,:);  %记录每一次迭代 每一架UAV的最优距离
    end
    %对不满足约束的个体进行替换
%     for p = 1:popSize
%         if popRoute(p,1)==0
%             popRoute(p,:)=Best_R;
%             popBreak(p,:)=Best_B;
%         end
%     end
    % Find the Best Route in the Population   
%     [min_dist,index] = min(totalDist);    %[M,I] = min(A)  M为具体值，I为对应下标
%     distHistory(iter) = min_dist;
    
    % 如果得到80次迭代满足约束的最优 路径基因型和中断基因型，将其赋给不满足约束的值
    
    
    
    
     
%     if min_dist < globalMin     %globalMin 一开始为无穷大
%     %若本次迭代时的最佳距离小于历史全局最小值。
%     %就把他画在图上，并记录一共画了几次。
%         globalMin = min_dist;
%         opt_rte = popRoute(index,:);     % 与下同理
%         opt_brk = popBreak(index,:);     % opt_brk 其实就是 Best_B
%         rng = [[1 opt_brk+1];[opt_brk n]]';
% 
%     end
 
   
    
% 子代个体的产生过程
% 产生一个随机序列，用于挑选随机的8个父代产生子代

    randomOrder = randperm(popSize);
    for p = 8:8:popSize
        rtes = popRoute(randomOrder(p-7:p),:);
        brks = popBreak(randomOrder(p-7:p),:);
        %随机挑选的8个父代   并找到其对应中断
        dists = totalDist(randomOrder(p-7:p));  %totalDist是经过80次迭代后  1*80 的历史值
        [ignore,idx] = min(dists); 
        %从这8个父代中挑选出最佳父代，用于产生8个子代。
        bestOf8Route = rtes(idx,:);
        bestOf8Break = brks(idx,:);
        routeInsertionPoints = sort(ceil(n*rand(1,2)));
        %从中挑选出基因序列的2个位置
        %这两个位置用来从父代中产生新的基因新的
        I = routeInsertionPoints(1);
        J = routeInsertionPoints(2);
        for k = 1:8 % Generate New Solutions
            tmpPopRoute(k,:) = bestOf8Route;
            tmpPopBreak(k,:) = bestOf8Break;
            switch k
                case 2 % Flip
                    %将最佳父代的基因型从上面两个位置中间的片段反转，产生一个子代。
                    tmpPopRoute(k,I:J) = tmpPopRoute(k,J:-1:I);
                case 3 % Swap
                    %交换这两个片段的基因，产生新子代。
                    tmpPopRoute(k,[I J]) = tmpPopRoute(k,[J I]);
                case 4 % Slide
                    % 自己看吧，描述不出
                    tmpPopRoute(k,I:J) = tmpPopRoute(k,[I+1:J I]);
                    %上面都是调整路径基因型的
                    %下面用于调整中断点基因型，过程差不多，大家可以自己看的     ***********中断点的调整？？？？？
                case 5 % Modify Breaks
                    %随机产生，跟最佳父代没关系的一代。
                    tmpPopBreak(k,:) = rand_breaks();
                case 6 % Flip, Modify Breaks
                    tmpPopRoute(k,I:J) = tmpPopRoute(k,J:-1:I);
                    tmpPopBreak(k,:) = rand_breaks();
                case 7 % Swap, Modify Breaks
                    tmpPopRoute(k,[I J]) = tmpPopRoute(k,[J I]);
                    tmpPopBreak(k,:) = rand_breaks();
                case 8 % Slide, Modify Breaks
                    tmpPopRoute(k,I:J) = tmpPopRoute(k,[I+1:J I]);
                    tmpPopBreak(k,:) = rand_breaks();
                otherwise % Do Nothing
            end
        end
        newPopRoute(p-7:p,:) = tmpPopRoute;
        newPopBreak(p-7:p,:) = tmpPopBreak;
    end
    popRoute = newPopRoute;
    popBreak = newPopBreak;
end

function breaks = rand_breaks()                  %function语法
                                                 %function [输出变量] = 函数名称(输入变量） 
    
    if minTour == 1 % No Constraints on Breaks   **对中断点没有限制  （最少经过城市为1）
        tmpBreaks = randperm(n-1);
        breaks = sort(tmpBreaks(1:nBreaks));      %sort 函数  按升序对向量进行排序       问题：此函数是取  randperm(n-1)  （n-1）各元素组成行向量并升序排列的前四个元素，为什么还要是（n-1）呢
    else % Force Breaks to be at Least the Minimum Tour Length   %强制中断至少为最小旅行长度
        num_adjust = find(rand < cumProb,1)-1;      %找到 0-1 任意数小于cumProb的第一个数
        spaces = ceil(nBreaks*rand(1,num_adjust));      %ceil函数：朝正无穷大方向取整
        adjust = zeros(1,nBreaks);
        for kk = 1:nBreaks
            adjust(kk) = sum(spaces == kk);
        end
        breaks = minTour*(1:nBreaks) + cumsum(adjust);                %  形成中断基因 
    end
end

%  sum_Length=[];
% for s = 1:nSalesmen
%    [r,~]=size(a2{s});
%    [shortest_Length,shortest_Route1 ]= ants(r,a2{s}); 
% %     disp(['区域',num2str(i),'第',num2str(s),'架无人机最短飞行路程为：']);
% %     disp( num2str(shortest_Length));
%     shortest_Route{s}=shortest_Route1;
%     sum_Length=[sum_Length, shortest_Length];
% end


end
