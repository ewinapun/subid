function [R_new] = qr_updating(R,y_new,beta)
% updating the QR matrix when new data comes in
j = size(R, 1) +1; 
[R_new] = myqrinsert( sqrt(beta)*R,j,y_new','row');
end

