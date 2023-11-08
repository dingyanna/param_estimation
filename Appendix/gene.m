ja = -10;
jb = 10;
lw = 2;
ms = 10;
fs = 12;
m = 10000;

x = linspace(ja, jb, m);
%fprintf("ind(1): %e\n", ind(1)); 
x1 = 6.854774439399799;
x2 = 12.922830384604826;
x3 = 16.94098067656294;

x1_1 = 6.854774439399799;
x2_1 = 12.922830384604826;
x3_1 = 22.956163995645916;

x1_2 = 5.82842712;
x2_2 = 10.90832691;
x3_2 = 34.97140521;
 
r1 = 7;
r2 = 13;
r3 = 17;

r1_1 = 7;
r2_1 = 13;
r3_1 = 23;
 

y = zeros(length(x),1);
y1 = zeros(length(x),1); 
disp(size(y))

for j = 1:length(x)
    y(j) = log((x3^x(j) + 1) / (x1^x(j) + 1)) ...
            - log(x3 / x1) ...
            * log((x2^x(j) + 1) / (x1^x(j) + 1)) ...
            / log(x2 / x1) ...
            - (log(r3 / r1) ...
            - log(x3 / x1) ...
            * log(r2 / r1) ...
            / log(x2 / x1));

    y1(j) = log((x3_1^x(j) + 1) / (x1_1^x(j) + 1)) ...
            - log(x3_1 / x1_1) ...
            * log((x2_1^x(j) + 1) / (x1_1^x(j) + 1)) ...
            / log(x2_1 / x1_1) ...
            - (log(r3_1 / r1_1) ...
            - log(x3_1 / x1_1) ...
            * log(r2_1 / r1_1) ...
            / log(x2_1 / x1_1));

end

data1 = zeros(length(x),3);
data1(:,1) = x;
data1(:,2) = y;
data1(:,3) = y1;
brn1 = array2table(data1);
% Save the table to a CSV file
writetable(brn1, 'gene_brn.csv'); 
% fig = figure('position', [100,100, 500, 300]);
% zeroLine = zeros(length(x));
% fprintf("y(1) %e y2(1) %e\n", y(1), y2(1));
% figure(1)
% clf
% hold on
% p1 = plot( x(1:m),y(1:m),'-', 'LineWidth', lw, 'MarkerSize', ms/2);
% p2 = plot( x(1:m),y1(1:m),'-', 'LineWidth', lw, 'MarkerSize', ms );
% plot( x(1:m), zeroLine(1:m),'k-', 'LineWidth', lw, 'MarkerSize', ms);
% xlim([0,2.2]);
% ylim([-0.001,0.001]);
% grid on 
% hold off
% 
% xlabel( '$h$', 'Interpreter','LaTeX' );
% ylabel( '$\phi$','Interpreter','LaTeX' );
% h = [p1(1);p2(1)];
% legend(h, '$\phi(h\ |\ r_1,r_2,r_3)$', '$\phi(h\ |\ r_1,r_2,r_4)$', 'Location', 'SouthEast','Interpreter','LaTeX');
% set(gca,'FontSize',fs);
% saveas(fig, 'gene.pdf', 'pdf');