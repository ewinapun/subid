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

clc; clear all; close all;
warning('off','all')
warning
%%

Tr = 5000; Ts = 500; k = 1; % dvide train-test sets
tstart = tic;
N = (Tr + Ts) * k; % number of samples
m = 0;l = 2; n = 2; % input/output/order
nmax = 3; % max order
i = 2*(nmax)/l;
L = Tr-(2*i);
beta = 0.99;
trial = 10;

ainit = [.6 .8]; % sorted eig ascendingly
b = zeros(n,m); c = eye(l,n); d = zeros(l,m);

% Ts_list = [500];
% for idx = 1:length(betalist)
for idx = 5
    if(idx == 1)
        type = 'non';beta = 1;
    elseif(idx == 2)
        type = 'auto';
    elseif(idx == 3)
        type = 'lin';    
    elseif(idx == 4)
        type = 'stepu';
    elseif(idx == 5)
        type = 'stepd';
    elseif(idx == 6)
        type = 'stlin';
    end
    disp(type);
    adiag = TV_matrix_A(ainit,Tr,Ts,1,type);

pe_base_all = nan(trial,n); pe_all = nan(trial,n); peadpt_all = nan(trial,n);
te_all = nan(trial,n); teadpt_all = nan(trial,n);
A_non_adpt = zeros(trial,n); D_all=nan(trial,L,n);
yp_all = nan(l,Ts,trial); ypyy_all= nan(l,Ts,trial);y_test_all = nan(l,Ts,trial);

for tt = 1:5
% generate the output
%     disp(' ');
    disp(['trial #',num2str(tt)]);
    mrand = randn(n+l); mu = (mrand + mrand')/2; cov = mu * mu';
    R1 = cov(1:n,1:n); R12 = cov(1:n,n+1:n+l); R2 = cov(n+1:n+l,n+1:n+l); 
    x = zeros(n,N+1);y = zeros(l,N);pe_non = []; pe_adpt = [];
%     y_test = zeros(l,Ts);
    for j=1:N
        a = diag(adiag(j,:));
        %Solve discrete-time algebraic Riccati equations
        % A'PA-E'PE-(A'XB+S)inv(B'PB+R)(B'PA+S')+Q = 0
        % [X,L,G] = dare(A,B,Q,R,S,E)
        [P,V,G] = dare(a',c',R1,R2,R12,eye(n)); 
        Kg = (a*P*c'+R12)*inv(c*P*c'+R2);    % Kalman gain
        L0 = c*P*c'+R2;    % cov of e(t)
        e = randn(1,l)*chol(L0);
%         x(:,j+1) = a * x(:,j)+ Kg * e';
%         y(:,j) = c * x(:,j)+e';
        x(:,j+1) = a * x(:,j) + randn(n,1);
        y(:,j) = c * x(:,j);
    end 
    y_train = y(:,1:Tr);y_test = y(:,Tr+1:Tr+Ts);
    y_test_all(:,:,tt) = y_test;
%% Running subspace id & obtaining prediction errors
%     disp('******* Non-adaptive Subspace Algorithm *******')
   order = n;
    %     disp(['Identifying order n = ',int2str(order)]);
        [A,~,C,~,K,R,~,ss] = subid(y_train,[],i,order,[],[],1);
        [yp_non,pe_non] = predic_ep(y_test,[],A,[],C,[],K);
        yp_all(:,:,tt) = yp_non';
%         pe(order-1,:) = pe;
        if(order == n);A_non_adpt(tt,:) = abs(eig(A));end
%     end 

%     baseline
    [yp_base,pe_base] = predic_ep(y_test,[],diag(adiag(Tr,:)),[],c,[],K);
    xhat = zeros(n,1);
    for t=1:Ts
         yp(t,:) = (C*xhat)';
         xhat = A * xhat + eye(n)*(y_test(:,t) - C*xhat);
    end
    yp(t+1,:) = (C*xhat);

    erv = ((y_test(:,1:end)')-yp(2:end,:));
    pe_base = sqrt(sum(erv.^2)./sum(y.^2))*100;
    % obtain PE of each trial
    pe_base_all(tt,:) = pe_base;
    pe_all(tt,:) = pe_non;
    
end

%% gather results
tElapsed = toc(tstart);
format bank
% obtain avg PE and TE across all trials
pe_base_avg = mean(pe_base_all,2) % avg between 2 outputs
sem_pe_base = std(pe_base_avg)/sqrt(trial);pe_base_avgMC = mean(pe_base_avg);

pe_non_avg = mean(pe_all,2) % avg between 2 outputs
sem_pe_non = std(pe_non_avg)/sqrt(trial);pe_non_avgMC = mean(pe_non_avg);
end