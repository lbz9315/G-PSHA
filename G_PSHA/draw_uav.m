function draw_uav(citys,shortest_Route)

figure(1)
for j=1:size(citys,2)
n=size(citys{j},1);
for ii=2:n
   plot3([citys{j}(shortest_Route{j}(ii-1),1),citys{j}(shortest_Route{j}(ii),1)],[citys{j}(shortest_Route{j}(ii-1),2),citys{j}(shortest_Route{j}(ii),2)],[citys{j}(shortest_Route{j}(ii-1),3),citys{j}(shortest_Route{j}(ii),3)],'-bo')
    hold on
end
grid on
for i = 1: n      %size(citys{j}, 1)
    text(citys{j}(i, 1), citys{j}(i, 2), ['   ' num2str(i)]);
end

end
text(citys{1}(shortest_Route{1}(1), 1), citys{1}(shortest_Route{1}(1), 2), '       起点');
text(citys{1}(shortest_Route{1}(end), 1), citys{1}(shortest_Route{1}(end), 2), '       终点');
end
