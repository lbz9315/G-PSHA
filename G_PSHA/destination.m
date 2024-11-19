

function [Z]=destination (center1,b,k)
% 得到每一个子区域中距离上一个中心点最近的送取点的位置
s2=zeros(1,k);

S1=[];
for i=1:k
    p=size(b{i},1);
    for j=1:p
        S1=[S1, ( b{i}(j,1)-center1(i,1) )^2 + ( b{i}(j,2)-center1(i,2) )^2];
        % 在每一个子区域里，求出每一个目标点与上个区域的距离的平方，将值存到向量 S 中
        
    end
    [~,s1]=sort(S1);   % s1 为按照从小到大排序后的对应值 下标 的序列   。则s1的第一位即在此子区域中，距离上一个中心点最近的点 的下标
    s2(i)=s1(1);    % 经过k次循环，s2中即为每一个子区域中，离上一个中心点最近的 目标点 的  下标
    S1=[];
    
end

Z=[];
for i=1:k
   Z=[Z;b{i}(s2(i),:)];
end
Z(:,3)=0;
end











