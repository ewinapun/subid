function [ sys_adpt, R_new ] = adaptive_train_onesteponly( data,k, R_past, nx, ny, horizon,beta )
%L = size(data,2)-(2*horizon); 
y_new = zeros(ny*horizon*2,1); 
%for k = 1 : L
    for k1 = 1 : 2*horizon
    y_new((k1-1)*ny+1 : k1*ny,1) = data(:,k+k1); 
    end
    [ R_new ] = qr_updating( R_past, y_new, beta );
    %
    %R_past=R_new;
    %disp(k); 
%end
%[sys_adpt] = subs_sysid_sto_qr(R_new, ny, nx, horizon, 'UPC');
[sys_adpt] = subs_sysid_sto_qr_parima(R_new, ny, nx, horizon, 'UPC');
end
