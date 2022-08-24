%% Function for calculating the deflection
function [def_val]=deflection_term(a,b,E,h,v,m,n,q,x,y)
                              % a,b,h are dimension of plate
                              % E is Youngs modulus,v is poissions ratio,q is UDL 
                              % m=n=3
                              % x,y are data
D=(E*h^3)/(12*(1-v));
def=0;
sum=0;
% Calculation of A and B for the calculation of deflection
for i =1:n
    val_n=i;
    for j=1:m
        val_m=j;
        A=sin((val_m.*pi.*x)./a).*sin((val_n.*pi.*y)./b);
        B=m*n*((val_m^2)./(a.^2)+(val_n^2)./(b.^2)).^2;
 
        % Deflection   
        def=(16*q.*A)./(pi^6*D.*B);
        sum=sum+def;
    end
end
def_val=sum;
end
