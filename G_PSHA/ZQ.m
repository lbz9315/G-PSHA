function [w]=ZQ(A2,J)
n=size(A2,1);
w=0;
for i=1:n
    s=num2str(A2(i,1));
    if isKey(J,s)
        w=w+J(s);
    end
end