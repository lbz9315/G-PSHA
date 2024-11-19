function [J]=jiaquan1(A2,B)
n=size(A2,1);
J=containers.Map();
for i=1:n
    s=num2str(A2(i,1));
    if ismember(A2(i,1),B)
%        J(s)=2.3*rand;
       J(s)=1+4*rand;
    else
       J(s)=0;
    end
end

