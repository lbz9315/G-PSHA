function D = Distance(citys)
%DISTANCE 计算两两城市之间的距离
 
n = size(citys,1);
D = zeros(n,n);
for i=1:n
    for j = i+1:n
        D(i,j) = sqrt(sum((citys(i,:) - citys(j,:)).^2));
        D(j,i) = D(i,j);%距离沿着对角线对称
    end
end