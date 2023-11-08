%% Prepare data

%% Set Parameters
B = 0.1;
K = 5;
C = 1;
D = 5;
E = 1; 

%% Get Steady States

% numLines = 2;
degrees = [7. 13. 17. 23. 29. 58. 68. 79. 89. 100.];
states = [];
for idegree = 1:length(degrees)
    beta = degrees(idegree);
    a = -  E / (C*K);
    b = ((C+K) * E - D) / (C*K);
    c = (C * (D - E * K + beta * K) + D * K) / (C*K);
    d = B * E - D;
    e = B * D;
    p = [a b c d e];
    rts = roots(p);

    real_rts = [];
    for i = 1:length(rts)
        if isreal(rts(i))
            real_rts(end+1) = rts(i);
        end
    end
    rt = max(real_rts);
    %fprintf('%e\n', rt);
    states(end+1) = rt;
end
display(states);
  
%% Get D-E relationship
tol1 = 2e-6; %1e-7
tol2 = 1e-5;

[Ds, Es] = plotEco(states(1), states(2), states(3), states(4), states(5), degrees(1), degrees(2), degrees(3), degrees(4), degrees(5), tol1, tol2);
fprintf('\n------\n');
[Ds_, Es_] = plotEco(states(6), states(7), states(8), states(9), states(10), degrees(6), degrees(7), degrees(8), degrees(9), degrees(10), tol1, tol1);
 
data1 = zeros(2,size(Ds,2));
data1(1,:) = Ds;
data1(2,:) = Es;
data2 = zeros(2,size(Ds_,2));
data2(1,:) = Ds_;
data2(2,:) = Es_;
brn1 = array2table(data1');
brn2 = array2table(data2');
% Save the table to a CSV file
writetable(brn1, 'brn1.csv');
writetable(brn2, 'brn2.csv');

%% Start Plotting
% clf 
% % Set Plot param
% lw = 2;
% ms = 12;
% fs = 12;
% fig = figure('position', [100, 100, 500, 400]); 
% 
% p1 = plot( Ds, Es,'o', 'LineWidth', lw, 'MarkerSize', ms*0.4 );
% hold on 
% p2 = plot( Ds_,Es_,'x', 'LineWidth', lw, 'MarkerSize', ms);
% 
% 
% ylim([0.9,1.05]);
% xlim([4.8,5.1]);
% grid on
% hold off
% xlabel( '$D$','Interpreter','LaTeX' );
% ylabel( "$E'$",'Interpreter','LaTeX' );
% h = [p1(1);p2(1)];
% legend(h, '$\Gamma_{[1,5]}$', '$\Gamma_{[6,10]}$', 'Location', 'SouthEast','Interpreter','LaTeX');
% set(gca,'FontSize',fs);
% 
% saveas(fig, 'eco.pdf', 'pdf');
% 