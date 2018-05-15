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

Tr = 5000; Ts = 0; k = 1; % dvide train-test sets
N = (Tr + Ts) * k; % length of inputs
beta = 1; 
trial = 2;
arrayAall = [];pe_all = []; peadpt_all = [];

% % creating a true system
% m = 2;l = 2;n = 4;
% [a1,a2,a3,a4] = TV_matrix_A(N,k);
% b = [1.1650,-0.6965;0.6268 1.6961;0.0751,0.0591;0.3516 1.7971];
% c = [0.2641,-1.4462,1.2460,0.5774;0.8717,-0.7012,-0.6390,-0.3600];
% d = zeros(l,m);

m = 1;l = 1; n = 2;
nmax = 5; % testing up to nmax
i = 2*(nmax)/l;
L = N-(2*i); 

ainit = [0.5 0.7];
adiag = TV_matrix_A(ainit,N,1,'non');
b = zeros(n,m); c = ones(l,n); d = zeros(l,m);
tstart = tic;
parfor tt = 1:trial
    disp(' ');
    disp(['trial #',num2str(tt)]);
    mrand = randn(n+l); mu = (mrand + mrand')/2; cov = mu * mu';
    R1 = cov(1:n,1:n); R12 = cov(1:n,n+1:n+l); R2 = cov(n+1:n+l,n+1:n+l); 

    % generate the output
    x = zeros(n,N+1);y = zeros(N,l);
    for j=1:N
        a = diag(adiag(j,:));
        %Solve discrete-time algebraic Riccati equations
        % A'PA-E'PE-(A'XB+S)inv(B'PB+R)(B'PA+S')+Q = 0
        % [X,L,G] = dare(A,B,Q,R,S,E)
        [P,V,G] = dare(a',c',R1,R2,R12,eye(n)); 
        Kg = (a*P*c'+R12)*inv(c*P*c'+R2);    % Kalman gain
        L0 = c*P*c'+R2;    % cov of e(t)
        e = randn(1,l)*chol(L0);
%         x(:,j+1) = a * x(:,j)+ b * u(j,:)'+Kg * e';
%         y(j,:) = (c * x(:,j)+ d * u(j,:)')' + e';
        x(:,j+1) = a * x(:,j)+ e';
        y(j,:) = (c * x(:,j))';
    end
    %% Running subspace id & obtaining prediction errors
    
    disp('******* Non-adaptive Subspace Algorithm *******')
    A_non_adpt = zeros(n,trial);
    for order = n
    %     disp(['Identifying order n = ',int2str(order)]);
        [A,B,C,D,K,R,~,ss] = subid(y,[],i,order,[],[],1);
        [yp,pe] = predic(y,[],A,[],C,[],K);
%         pe(order-1,:) = pea;
        if(order == n);A_non_adpt(:,tt) = abs(eig(A));end
    end 
    pe_all = [pe_all,pe];
    
    disp('*******   Adaptive Subspace Algorithm   *******');
    
    data = y'; 
    [ R_past ] = qr_initialization( data, i, beta ); 
    y_new = zeros(l*i*2,1); 
    poles = zeros(n,L); 
    for order = n
        %     disp(['Identifying order n = ',int2str(order)]);
        for k = 1 : L
            for k1 = 1 : 2*i
                y_new((k1-1)*l+1 : k1*l,1) = data(:,k+k1-1);
            end
            [ R_new ] = qr_updating( R_past, y_new, beta );
%             [as] = adaptive_gamma_subid(R_new, l, order, i, 'UPC', beta);
            [as] = subs_sysid_sto_qr_parima(R_new, l, order, i, 'UPC');
%             [as.a,~,as.c,~,as.k,~,~,~] = subid(y(1:k,:),[],i,order,[],[],1);
%             if(order == n)
                poles(:,k) = sort(abs(eig(as.a)));
%             end
            R_past = R_new;
        end
        [ypyy,peadpt] = predic(y,[],as.a,[],as.c,[],as.k);
%         peadpt(order-1,:) = pea;
    end
    peadpt_all = [peadpt_all,peadpt];
    poles = poles';
    for ii = 1:n
        arrayAall(:,tt,ii) = poles(:,ii);
    end
end

arrayAavg = [];temp = [];A_non_adpt_avg = [];
for ii = 1:n
    temp = arrayAall(:,:,ii);
    arrayAavg(:,ii) = (sum(temp')/trial)';
%     A_non_adpt_avg(ii,1) = sum(A_non_adpt(ii,:));
    clear temp
end
pe_avg = (sum(pe_all')/trial)'
pe_adpt_avg = (sum(peadpt_all')/trial)'
tElapsed = toc(tstart)
%% plot
% if l == 2
%     figure(2);
%     subplot(211)
%     plot([1:nmax],pe_avg(:,1),'k','LineWidth',3);hold on;
%     plot([1:nmax],pe_adpt_avg(:,1),'--k','LineWidth',3)
%     legend('non-adaptive sub id','adaptive sub id')
%     title('Averaged Prediction Error of Output 1 vs system order');
%     xlabel('system order');ylabel('Prediction Error (%)');
%     axis([0,nmax+1,0,100]);
%     subplot(212)
%     plot([1:nmax],pe_avg(:,2),'k','LineWidth',3);hold on;
%     plot([1:nmax],pe_adpt_avg(:,2),'--k','LineWidth',3)
%     legend('non-adaptive sub id','adaptive sub id')
%     title('Averaged Prediction Error of Output 2 vs system order');
%     xlabel('system order');ylabel('Prediction Error (%)');
%     axis([0,nmax+1,0,100]);
% end
% if l == 1
%     figure(2);
%     plot([1:nmax],pe_avg(:,1),'k','LineWidth',3);hold on;
%     plot([1:nmax],pe_adpt_avg(:,1),'--k','LineWidth',3)
%     legend('non-adaptive sub id','adaptive sub id')
% %     for tt = 1:trial
% %         plot([1:nmax],peall(:,tt),'r')
% %         plot([1:nmax],peadptall(:,tt),'c')
% %     end
%     title('Averaged Prediction Error of output vs system order');
%     xlabel('system order');ylabel('Prediction Error (%)');
%     axis([1,nmax,0,100]);
% end

figure()
for ii = 1:n
    plot(arrayAavg(:,ii),'LineWidth',0.5);hold on;
end
plot([1:L],adiag(1:L,:),'--k');
legend('estimated eig(A) 1','estimated eig(A) 2','true eig(a) 1','true eig(a) 2','Location','southeast')
title('averaged change in eiganvalue of A')
axis([1,L,0,1]);

figure()
plot([1:L],adiag(1:L,:),'--k');hold on
for tt = 1:trial
        plot(arrayAall(:,tt,1),'r')
        plot(arrayAall(:,tt,2),'c')
end
legend('true eig(a) 1','true eig(a) 2','estimated eig(A) 1','estimated eig(A) 2','Location','southeast')
title('averaged change in eiganvalue of A')
axis([1,L,0,1]);


