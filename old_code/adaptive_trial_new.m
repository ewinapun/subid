% Test for Stochastic Subspace identification
%
% a is a nxn matrix
% b is a nxm matrix
% c is a lxn matrix
% d is a lxm matrix
% u is a m-dimensional vector input
% x is a n-dimensional vector state
% y is a l-dimensional vector output

% With noise added, the state space system equations become:
%  x_{k+1) = A x_k + B u_k + w_k (= K *e(k) )       
%    y_k   = C x_k + D u_k + v_k (= e(k) )
% cov(v_k) = R
% where v_k is l dimension, w_k is n dimension, and R is lxl dimension

clc; clear all; close all;
warning('off','all')
warning

%% part 1: create the system

Tr = 2000; Ts = 0; k = 1; % dvide train-test sets
N = (Tr + Ts) * k; % length of inputs

% creating a true system
m = 2;l = 2;n = 4;
ainit = [0.9 0.85 0.8 0.75];
[adiag] = TV_matrix_A(ainit,N,k,0); %TV_matrix_A(a,N,k,autoregressive) 
% a = [0.603 0.603 0 0;-0.603 0.603 0 0;0 0 -0.603 -0.603;0 0 0.603 -0.603];
% b = [1.1650,-0.6965;0.6268 1.6961;0.0751,0.0591;0.3516 1.7971];
b = zeros(n,m);
c = [0.2641,-1.4462,1.2460,0.5774;0.8717,-0.7012,-0.6390,-0.3600];
d = zeros(l,m);
u = randn(N,m);

% m = 1;l = 1; n = 2;
% a = diag([0.8 0.9]);
% b = [1.1650;0.6268];
% c = [0.2641 0.5774];
% d = [0];
% u = randn(N,m);

mrand = 0.1*randn(n+l); mu = (mrand + mrand')/2; cov = mu * mu';
R1 = cov(1:n,1:n); R12 = cov(1:n,n+1:n+l); R2 = cov(n+1:n+l,n+1:n+l); 

% generate the output y
x(:,1)=zeros(n,1); eall = zeros(N,l);
for j=1:N
    a = diag(adiag(j,:));
    %       Solve discrete-time algebraic Riccati equations
    %       A'PA-E'PE-(A'XB+S)inv(B'PB+R)(B'PA+S')+Q = 0
    %       [X,L,G] = dare(A,B,Q,R,S,E)
    [P,V,G] = dare(a',c',R1,R2,R12,eye(n)); 
    Kg = (a*P*c'+R12)*inv(c*P*c'+R2);    % Kalman gain
    L = c*P*c'+R2;    % cov of e(t)
    e = randn(1,l)*chol(L);
    x(:,j+1) = a * x(:,j)+b * u(j,:)'+Kg * e';
    y(j,:) = (c * x(:,j)+d * u(j,:)'+e')';
    eall(j,:) = e;
end
clear j

%  ynow = dlsim(a,b,c,d,u) + dlsim(a,Kg,c,eye(1),eall);

%%

nmax = 6; % testing up to nmax
i = 2*(nmax)/l;
disp('******* Non-adaptive Subspace Algorithm *******')
[A,B,C,D,K,R,~,ss] = subid(y,[],i,n,[],[],1);
[yp,pe] = predic(y,[],A,[],C,[],K);
pe
%  
% [A,B,C,D,K,R,ss] = subid(y,u,i,n,[],[],1);
% [yp,pe] = predic(y,u,A,B,C,D,K);
% pe

% for order = 1:nmax
%     disp(['Identifying order n = ',int2str(order)]);
% %     [A,B,C,D,K,R,ss,pea] = traintestfun(y,u,i,Tr,Ts,k,nmax,order);
% %     if(k~=1)
% %         pe(order,:) = mean(pea);
% %     else
%     [A,B,C,D,K,R,~,ss] = subid(y,[],i,order,[],'CVA',0);
%     [yp,pe] = predic(y,[],A,[],C,[],K);
%     pea(order,:) = pe;
% %     end
% end
% pea

%% 
disp('******* Adaptive Subspace Algorithm *******');
% modified gamma method
data = y'; 
beta = 0.999; 
[ R_past ] = qr_initialization( data, i, beta ); 
L = size(data,2)-(2*i); 
y_new = zeros(l*i*2,1); 
poles = zeros(n,L); 
for k = 1 : L
    for k1 = 1 : 2*i
        y_new((k1-1)*l+1 : k1*l,1) = data(:,k+k1-1);
    end
    [ R_new ] = qr_updating( R_past, y_new, beta );
    [as] = adaptive_gamma_subid(R_new, l, n, i, 'UPC', beta);
    R_past = R_new;
    poles(:,k) = abs(eig(as.a));
end
[ypyy,peyy] = predic(y,[],as.a,[],as.c,[],as.k);
peyy
