arrayA = [];

for i = 1:length(allsyscomb)
    aa = allsyscomb{i,n};
    [V,D] = eig(aa.A);
    D = real(D);
    arrayA = [arrayA;diag(D)'];
end

figure()
for ii = 1:n
    plot(arrayA(:,ii),'r');hold on;
    legend('estimated eig(A)')

end
plot([1:length(allsyscomb)],adiag(1:length(allsyscomb),:),'b');
legend('true eig(a) 1','true eig(a) 2','estimated eig(A) 1','estimated eig(A) 2','Location','southeast')
title('change in eiganvalue of A')
