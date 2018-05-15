data = y'; 
i = 20; 
beta = 1; 
data_train = data; 
[ R_past ] = qr_initialization( data_train, i, beta ); 
nx = n; 
ny = l;
L = size(data,2)-(2*i); 
y_new = zeros(ny*i*2,1); 
poles = zeros(nx,L); 
for k = 1 : L
    for k1 = 1 : 2*i
        y_new((k1-1)*ny+1 : k1*ny,1) = data(:,k+k1); 
    end
    [ R_new ] = qr_updating( R_past, y_new, beta );
    R_past = R_new;
   [sys_adpt] = subs_sysid_sto_qr(R_new, ny, nx, i, 'UPC');
    poles(:,k) = abs(eig(sys_adpt.a)); 
%     disp(k); `
end

figure()
for ii = 1:n
    plot(poles(ii,:));hold on;
end
plot([1:L],adiag(1:L,:),'--k');
legend('estimated eig(A) 1','estimated eig(A) 2','true eig(a) 1','true eig(a) 2','Location','southeast')
title('averaged change in eiganvalue of A')
axis([1,L,0,1]);