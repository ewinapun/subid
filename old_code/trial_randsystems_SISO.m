clc; clear all; close all;

%% part 1: create the system
% a is a nxn matrix
% b is a nxm matrix
% c is a lxn matrix
% d is a lxm matrix
% u is a m-dimensional vector input
% x is a n-dimensional vector state
% y is a l-dimensional vector output
% first simulate a SISO system of order n = 4:

% With noise added, the state space system equations become:
%  x_{k+1) = A x_k + B u_k + w_k (= K *(y_k-c*x_k))       
%    y_k   = C x_k + D u_k + v_k (=)
% cov(v_k) = R
% where v_k is l dimension, w_k is n dimension, and R is lxl dimension


m = 1; l = 1;
N = 80000; % length of inputs
n = 2;
a_waveform = 0.5*ones(N,1); % constant
a = [0.603  0; 0 0.803];
b = [1.1650;0.6268];
c = [0.2641 0.5774];
d = [0];
u = randn(N,m);
mrand = 0.05*randn(n+l);mu = (mrand + mrand')/2; cov = mu * mu';
R1 = cov(1:n,1:n); R12 = cov(1:n,n+1:n+l); R2 = cov(n+1:n+l,n+1:n+l); 

%Solve discrete-time algebraic Riccati equations
% A'PA-E'PE-(A'XB+S)inv(B'PB+R)(B'PA+S')+Q = 0
% [X,L,G] = dare(A,B,Q,R,S,E)
[P,V,G] = dare(a',c',R1,R2,R12,eye(n)); 
K = (a*P*c'+R12)*inv(c*P*c'+R2);    % Kalman gain
L = c*P*c'+R2;    % cov of e(t)
e = randn(N,l)*chol(L);
% x(:,1)=zeros(length(a),1); y=zeros(N,m);
% for k=1:N
%     x(:,k+1) = a*x(:,k)+b*u(k,m)+K*e(k);
%     y(k,m) = c*x(:,k)+d*u(k,m)+e(k);
% end

% K = [0.1242;-0.0828;0.0390;-0.0225];
% r = [0.05];   % A positive definite matrix is a symmetric matrix with all positive eigenvalues
%                 % square of any symmetric (to get +ve eiganvalue)
% e = randn(N,1)*chol(r);
% % implementing own dlsim(a,b,c,d,u)
% x(:,1)=zeros(n_order,1);
% y=zeros(N,1);
% for k=1:N
%     x(:,k+1) = a_waveform(k)*a*x(:,k)+b*u(k,m)+K*e(k);
%     y(k,m) = c*x(:,k)+d*u(k,m)+e(k);
% end
 y = dlsim(a,b,c,d,u)+ dlsim(a,K,c,eye(1),e);
%% Running subspace id & obtaining prediction errors in different scenario
nmax = 10; % testing up to nmax
i = 2*(nmax)/1; % number of outputs is 1
[A,B,C,D,K,R,ss] = Ewina_subid(y,u,i,4,[],1);
[yp,pe] = predic(y,u,A,B,C,D,K);
disp(['PE =', num2str(pe), ' with order = ', num2str(n)]);

figure(gcf);
H = bar([1:l*i],ss); xx = get(H,'XData'); yy = get(H,'YData'); 
semilogy(xx,yy+10^(floor(log10(min(ss))))); % log scale on y-axis
axis([0,length(ss)+1,10^(floor(log10(min(ss)))),10^(ceil(log10(max(ss))))]);
title('Singular Values');
xlabel('Order');

figure
subplot(311);plot(u(:,m));title('Input u');
subplot(312);plot(y(:,l));title('Output y with noise');
subplot(313);plot(y(:,l));title('Predicted output');
hold on
plot(yp(:,l));
legend('original output','predicted output');

Tr = 2000; Ts = 2000; k = 20;
for order = 1:nmax
    disp(['Identifying order n = ',int2str(order)]);
    [A,B,C,D,K,R,ss,pea] = traintestfun(y,u,i,Tr,Ts,k,nmax,order);
    pe(order,:) = mean(pea);
end
figure
bar([1:nmax],pe);
title('pe vs system order');
xlabel('System order');
axis([0,nmax+1,0,max(pe)+10]);