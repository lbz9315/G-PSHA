function points = Equal_parts(posA,posB,n)

% points=[];
% posA=[0,0,3];
% posB=[3,2,9];
% posA=[0,0];
% posB=[3,2];
% n=12;
% points=posA;
% for i = n-1:-1:1
%    k=1/i;
%    temp=(posA+k*posB)/(1+k);
%    posA = temp;
%    points=[points;temp];
% end
% points = [points;posB];
% % plot3(points(:,1),points(:,2),points(:,3),'ro-')
% plot(points(:,1),points(:,2),'ro-')


% function points = Equal_parts(posA,posB,n)
points=[];
for i = n-1:-1:1
 k=1/i;
 temp=(posA+k*posB)/(1+k);
 posA = temp;
 points=[points;temp];
end
% end