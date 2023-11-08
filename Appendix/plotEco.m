function [Ds, Es]=plotEco(x1, x2, x3, x4, x5, ...
                     r1, r2, r3, r4, r5, tol, tol2)
    %% set denominators
    denom1 = (x2 - x3) * ...
             (-x1 + x3) * ...
             (-x1^2 * x2 + x1 * x2^2 - x1^2 * x3 + x2^2 * x3);
    denom2 = (x1 - x2) * ...
             (x1 - x3) * ...
             (x2 - x3) * ...
             (x1 * x2 + x1 * x3 + x2 * x3);
    denom3 = (x1 - x2) * ...
             (x1 - x3) * ...
             (x2 - x3) * ...
             (x1 * x2 + x1 * x3 + x2 * x3);
     
    %% set coefficients
    a1 = getCoeff(x2, x3, x4, denom1, denom2, denom3);
    a2 = getCoeff1(x1, x3, x4, denom1, denom2, denom3);
    a3 = getCoeff(x1, x2, x4, denom1, denom2, denom3);
    
    b1 = getCoeff(x2, x3, x5, denom1, denom2, denom3);
    b2 = getCoeff1(x1, x3, x5, denom1, denom2, denom3);
    b3 = getCoeff(x1, x2, x5, denom1, denom2, denom3);
    
    
    m  = 5000;
    D1  = linspace(3,10, m);
    E1  = linspace(0.5,2, m);  
    
    D = zeros(1, size(D1,2)+1);
    D(1:end-1) = D1;
    D(end) = 5;

    E = zeros(1, size(E1,2)+1);
    E(1:end-1) = E1;
    E(end) = 1;

    %D = [3.5, 3.6, 3.7, 3.8, 3.9, 4, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5];
    %E = [0.7,0.8,0.9,1,1.1];

    z1 = zeros(size(D,2),size(E,2));
    z2 = zeros(size(D,2),size(E,2));

    disp(size(D,2))
    
    Ds = [];
    Es = [];
    for iD = 1:size(D,2)
        for iE = 1:size(E,2) 
            z1(iD, iE) = getVal(a1, a2, a3, ...
                              r1, r2, r3, r4, ...
                              x1, x2, x3, x4, D(iD), E(iE));

            z2(iD, iE) = getVal(b1, b2, b3, ...
                              r1, r2, r3, r5, ...
                              x1, x2, x3, x5, D(iD), E(iE));
             
            if abs(z1(iD, iE)) < tol2 && abs(z2(iD, iE)) < tol
                fprintf("%d %d\n", iD, iE);
                fprintf("D(iD) %.3f   E(iE)  %.3f\n", D(iD), E(iE));
                fprintf("abs(z1(iD, iE)) %e\n", abs(z1(iD, iE)));
                fprintf("abs(z2(iD, iD)) %e\n", abs(z2(iD, iE)));
                Ds(end+1) = D(iD);
                Es(end+1) = E(iE); 
            end
        end
    end


    %% Start Plotting
%     clf
%     hold on  
    %% Set Plot param
%     lw = 2;
%     ms = 16;
%     fs = 20;

    %plot( Ds,Es,'k-', 'LineWidth', lw, 'MarkerSize', ms, 'Color', '#0072BD');

    %% Surface Plot
    %surf( D,E,z1.', 'FaceColor', '#0072BD' ,'EdgeColor', 'none')
    %alpha(.5)
    %surf( D,E,z2.', 'FaceColor', '#D95319', 'EdgeColor', 'none')
    %alpha(.7)
    %surf( D,E,z.', 'FaceColor', 'green', 'EdgeColor', 'none')
    
    %xlabel( 'D' );
    %ylabel( 'E' );
    %set(gca,'FontSize',fs);
end

function [z] = getVal(a1, a2, a3, ...
                      r1, r2, r3, r4, ...
                      x1, x2, x3, x4, D, E)
                  
    z = - a1 * r1 * x1^2 / (D + E * x1) ...
        - a2 * r2 * x2^2 / (D + E * x2) ...
        - a3 * r3 * x3^2 / (D + E * x3) ...
        + r4 * x4^2 / (D + E * x4) ...
        - x4 + a1 * x1 + a2 * x2 + a3 * x3;
    return
end

function [a] = getCoeff(x1, x2, x, denom1, denom2, denom3)
    term1 = (x1^3 * x2^2 - x1^2 * x2^3) / denom1;
    term2 = (x2^3 - x1^3) / denom2;
    term3 = (x2^2 - x1^2) / denom3;
    
    a = term1 + x^2 * term2 - x^3 * term3;
    return
end

function [a] = getCoeff1(x1, x2, x, denom1, denom2, denom3)
    term1 = (x1^2 * x2^3 - x1^3 * x2^2) / denom1;
    term2 = (x1^3 - x2^3) / denom2;
    term3 = (x1^2 - x2^2) / denom3;
    
    a = term1 + x^2 * term2 - x^3 * term3;
    return
end