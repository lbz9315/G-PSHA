function DrawPath(Chrom, X)
%% ��·��ͼ����
% ���룺
% Chrom������·��
% X����Ŀ�������λ��
% R = [Chrom(1, :) Chrom(1, 1)];  % һ������⣨���壩    Chrom(1, 1)��ʾ�ְ��������ˣ��Ӷ��γɱ�·
R =Chrom(1, :);
% figure;
hold on
plot3(X(:, 1), X(:, 2),X(:,3) ,'o', 'color', [0.5, 0.5, 0.5])
plot3(X(Chrom(1, 1), 1), X(Chrom(1, 1), 2), X(Chrom(1, 1), 3), 'rv', 'MarkerSize', 20)      %�����Ƿ�����ʾ��ʼ��

for i = 1: size(X, 1)
    text(X(i, 1) + 0.05, X(i, 2) + 0.05, num2str(i), 'color', [1, 0, 0]);     %�����ĵ���
end
A = X(R, :);            %A�ǽ�֮ǰ������˳����R���Һ����´���A��
row = size(A, 1);          %rowΪ������+1

%��Ϊ��ά�ģ�Ϊz�ḳֵ
arrowz=zeros(2,1);

for i = 2: row
  
%     [arrowx, arrowy] = dsxy2figxy( gca, A(i - 1: i, 1), A(i - 1: i, 2));        %dsxy2figxy����ת����������¼������
    %  annotation('textarrow', arrowx, arrowy, 'HeadWidth', 8, 'color', [0, 0, 1]);   %�����������ô���ͷ���߶��������� 
  plot3( A(i - 1: i, 1), A(i - 1: i, 2),arrowz,'-k');
end
view(3)
% hold off
xlabel('X(m)')
ylabel('Y(m)')
zlabel('Z(m)')
% title('�켣ͼ')
box on  