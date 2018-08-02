clc; clear all; close all;
warning('off','all')
warning
%% part 1: create the system
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

Tr = 1000; Ts = 1000; k = 1; % dvide train-test sets
N = (Tr + Ts) * k; % length of inputs

% creating a true system
% m = 2;l = 2;n = 4;
% [a1,a2,a3,a4] = TV_matrix_A(N,k);
% b = [1.1650,-0.6965;0.6268 1.6961;0.0751,0.0591;0.3516 1.7971];
% c = [0.2641,-1.4462,1.2460,0.5774;0.8717,-0.7012,-0.6390,-0.3600];
% d = zeros(l,m);
% u = randn(N,m);

m = 1;l = 1; n = 2;
ainit = [0.7 0.9];
adiag = TV_matrix_A(ainit,N,k,1);
b = [1.1650;0.6268];
c = [0.2641 0.5774];
d = [0];
u = randn(N,m);
mrand = 0.1*randn(n+l); mu = (mrand + mrand')/2; cov = mu * mu';
R1 = cov(1:n,1:n); R12 = cov(1:n,n+1:n+l); R2 = cov(n+1:n+l,n+1:n+l); 

% generate the output
x(:,1)=zeros(n,1); eall = zeros(N,l);
for j=1:N
    a = diag(adiag(j,:));
    %Solve discrete-time algebraic Riccati equations
    % A'PA-E'PE-(A'XB+S)inv(B'PB+R)(B'PA+S')+Q = 0
    % [X,L,G] = dare(A,B,Q,R,S,E)
    [P,V,G] = dare(a',c',R1,R2,R12,eye(n)); 
    Kg = (a*P*c'+R12)*inv(c*P*c'+R2);    % Kalman gain
    L = c*P*c'+R2;    % cov of e(t)
    e = randn(1,l)*chol(L);
    x(:,j+1) = a * x(:,j)+b * u(j,:)'+Kg * e';
    y(j,:) = (c * x(:,j)+d * u(j,:)'+e')';
    eall(j,:) = e;
end
clear i j
%  ynow = dlsim(a,b,c,d,u) + dlsim(a,Kg,c,eye(1),eall);
%% Running subspace id & obtaining prediction errors

nmax = 5; % testing up to nmax
i = 2*(nmax)/l; % number of outputs is 1
% [A,B,C,D,K,R,ss] = Ewina_subid(y,u,i,n,[],1);
% [yp,pe] = predic(y,u,A,B,C,D,K);

disp('******* Non-adaptive Subspace Algorithm *******')
for order = 1:nmax
%     disp(['Identifying order n = ',int2str(order)]);
    [A,B,C,D,K,R,ss,pea] = traintestfun(y,u,i,Tr,Ts,k,nmax,order);
    if(k~=1)
        pe(order,:) = mean(pea);
    else
        pe(order,:) = pea;
    end
end 
pe

allsyscomb = cell(Tr-2*i+1,nmax);
disp('******* Adaptive Subspace Algorithm *******');
for order = 1:nmax
%     disp(['Identifying order n = ',int2str(order)]);
    [pea,allsys] = adaptive_traintest(y,u,i,Tr,Ts,k,nmax,order);
    allsyscomb(:,order) = allsys;
    peadpt(order,:) = mean(pea);
end
peadpt

%% plot
if l == 2
    figure(2);
    subplot(211)
    plot([1:nmax],pe(:,1),'b');hold on;plot([1:nmax],peadpt(:,1),'--r')
    legend('non-adaptive sub id','adaptive sub id')
    title('Prediction Error of Output 1 vs system order');
    xlabel('System order');ylabel('Prediction Error');
    axis([0,nmax+1,0,100]);
    subplot(212)
    line([1:nmax],pe(:,2));line([1:nmax],peadpt(:,2),'--r')
    title('Prediction Error of Output 2 vs system order');
    xlabel('System order');ylabel('Prediction Error');
    axis([0,nmax+1,0,100]);
end
if l == 1
    figure(2);
    plot([1:nmax],pe(:,1),'b');hold on;plot([1:nmax],peadpt(:,1),'--r')
    legend('non-adaptive sub id','adaptive sub id')
    title('Prediction Error of Output 1 vs system order');
    xlabel('System order');ylabel('Prediction Error');
    axis([1,nmax,0,100]);
end

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
