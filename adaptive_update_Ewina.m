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
betalist = [1];
trial = 8;

ainit = [.6 .8];
ainit = sort(ainit);
b = zeros(n,m); c = eye(l,n); d = zeros(l,m);

% Ts_list = [500];
% for idx = 1:length(betalist)
for idx = 1
    beta = betalist(idx);
    if(idx == 1)
        type = 'non';
    elseif(idx == 2)
        type = 'auto';
    elseif(idx == 3)
        type = 'lin';    
    elseif(idx == 4)
        type = 'stepd';
    elseif(idx == 5)
        type = 'stlin';
    elseif(idx == 6)
        type = 'stepu';
    end
%     type = 'stepu';
    adiag = TV_matrix_A(ainit,Tr,Ts,1,type);

pe_base_all = nan(trial,n); pe_all = nan(trial,n); peadpt_all = nan(trial,n);
te_all = nan(trial,n); teadpt_all = nan(trial,n);
A_non_adpt = zeros(trial,n); D_all=nan(trial,L,n);
yp_all = nan(l,Ts,trial); ypyy_all= nan(l,Ts,trial);y_test_all = nan(l,Ts,trial);

for tt = 1:trial
% generate the output
%     disp(' ');
    disp(['trial #',num2str(tt)]);
    mrand = 0.1*randn(n+l); mu = (mrand + mrand')/2; cov = mu * mu';
    R1 = eye(n); R12 = zeros(n,l); R2 = 0.0001*eye(l);
    %R1 = cov(1:n,1:n); R12 = cov(1:n,n+1:n+l); R2 = cov(n+1:n+l,n+1:n+l); 
    x = zeros(n,N+1);y = zeros(l,N);
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
        x(:,j+1) = a * x(:,j) + Kg * e';
        y(:,j) = c * x(:,j)+e';
    end 
    y_train = y(:,1:Tr);y_test = y(:,Tr+1:Tr+Ts);
    y_test_all(:,:,tt) = y_test;
%% Running subspace id & obtaining prediction errors
%     baseline
    [yp_base,pe_base] = predic_ep(y_test,[],diag(adiag(Tr,:)),[],c,[],Kg);

%     disp('******* Non-adaptive Subspace Algorithm *******')
   order = n;
    %     disp(['Identifying order n = ',int2str(order)]);
        [A,~,C,~,K,R,~,Qs,Rs,Ss] = subid(y_train,[],i,order,[],[],1);
        [yp_non,pe_non] = predic_ep(y_test,[],A,[],C,[],K);
        yp_all(:,:,tt) = yp_non';
%         pe(order-1,:) = pe;
        if(order == n);A_non_adpt(tt,:) = sort(abs(eig(A)));end
%     end 

%     disp('*******   Adaptive Subspace Algorithm   *******');
    
    [ R_past ] = qr_initialization( y_train, i, beta ); 
    y_new = zeros(l*i*2,1); 
    sys_adpt_overtime=[];
    
%     for order = n
        %     disp(['Identifying order n = ',int2str(order)]);
        for k = 1 : L
            for k1 = 1 : 2*i
                y_new((k1-1)*l+1 : k1*l,1) = y_train(:,k+k1-1);
            end
            [ R_new ] = qr_updating( R_past, y_new, beta );
%             [as] = adaptive_gamma_subid(R_new, l, order, i, 'UPC', beta);
            [as] = subs_sysid_sto_qr_Ewina(R_new, l, order, i, 'UPC', beta);
            R_past = R_new;
            sys_adpt_overtime=[sys_adpt_overtime;as];
        end
        [yp_adpt,pe_adpt] = predic_ep(y_test,[],as.a,[],as.c,[],as.k);
%         pe_adpt(order-1,:) = pea;
        ypyy_all(:,:,tt) = yp_adpt';
%     end

    % obtain PE of each trial
    pe_base_all(tt,:) = pe_base;
    pe_all(tt,:) = pe_non;
    peadpt_all(tt,:) = pe_adpt;
    
    % obtain TE of each trial
    D = zeros(L,n,n); D_sorted = zeros(L,n);
    for k=1:L
        [~,D(k,:,:)] = eig(sys_adpt_overtime(k).a);
        D_sorted(k,:) = sort(diag(abs(squeeze(D(k,:,:)))));
    end
    for ii = 1:n
        D_all(tt,:,ii) = D_sorted(:,ii);
    end
    erv = A_non_adpt(tt,:)- adiag(1:L,:);
    te_non = sqrt(sum(erv.^2)./sum(adiag(1:L,:).^2))*100;
    erv_adpt = D_sorted(1:L,:) - adiag(1:L,:);
    te_adpt = sqrt(sum(erv_adpt.^2)./sum(adiag(1:L,:).^2))*100;
    te_all(tt,:) = te_non;
    teadpt_all(tt,:) = te_adpt;
end

%% gather results
tElapsed = toc(tstart);

% obtain avg PE and TE across all trials
pe_base_avg = mean(pe_base_all,2); % avg between 2 outputs
sem_pe_base = std(pe_base_avg)/sqrt(trial);pe_base_avgMC = mean(pe_base_avg);

pe_non_avg = mean(pe_all,2); % avg between 2 outputs
sem_pe_non = std(pe_non_avg)/sqrt(trial);pe_non_avgMC = mean(pe_non_avg);

pe_adpt_avg = mean(peadpt_all,2); % avg between 2 outputs
sem_pe_adpt = std(pe_adpt_avg)/sqrt(trial);pe_adpt_avgMC = mean(pe_adpt_avg);

te_non_avg = mean(te_all,2); % avg between 2 outputs
sem_te_non = std(te_non_avg)/sqrt(trial);te_non_avgMC = mean(te_non_avg);

te_adpt_avg = mean(teadpt_all,2); % avg between 2 outputs
sem_te_adpt = std(te_adpt_avg)/sqrt(trial);te_adpt_avgMC = mean(te_adpt_avg);

p = signrank(pe_non_avg,pe_adpt_avg);

% for plotting averaged poles
D_avg = zeros(L,n);
for ii = 1:n
    D_avg(:,ii) = nanmean(abs(D_all(:,:,ii)));
end
A_non_avg = mean(A_non_adpt);
erv2 = A_non_avg- adiag(1:L,:);
te_non2 = sqrt(sum(erv2.^2)./sum(adiag(1:L,:).^2))*100;
erv_adpt2 = D_avg(1:L,:) - adiag(1:L,:);
te_adpt2 = sqrt(sum(erv_adpt2.^2)./sum(adiag(1:L,:).^2))*100;

disp(type);
disp([num2str(pe_base_avgMC),char(177),num2str(sem_pe_base)]);
disp([num2str(pe_non_avgMC),char(177),num2str(sem_pe_non)]);
disp([num2str(pe_adpt_avgMC),char(177),num2str(sem_pe_adpt)]);
disp([num2str(te_non_avgMC),char(177),num2str(sem_te_non)]);
disp([num2str(te_adpt_avgMC),char(177),num2str(sem_te_adpt)]);
disp(p);
disp([mean(te_non2),std(te_non2)]);
disp([mean(te_adpt2),std(te_adpt2)]);

% for plotting estimation of test set
[~,idxy] = max(pe_all-peadpt_all);
y_test = y_test_all(:,:,idxy);
yp = yp_all(:,:,idxy);
ypyy = ypyy_all(:,:,idxy);

%% export data to spreadsheet
filename = ['data-stepu-05-09-2018-beta'];
sheet = [type,'-beta',num2str(beta)];
pewrite = {'pe_baseline 1','pe_baseline 2','pe_base_avg',...
    'pe_non-adpt 1','pe_non-adpt 2','pe_non-adpt_avg',...
    'pe_adpt 1','pe_adpt 2','pe_adpt_avg',...
    'te_non-adpt 1','te_non-adpt 2','te_non-adpt_avg',...
    'te_adpt 1','te_adpt 2','te_adpt_avg'};
xlswrite(filename,pewrite,sheet,'A1');
write = [pe_base_all(:,1),pe_base_all(:,2),pe_base_avg,pe_all(:,1),pe_all(:,2),pe_non_avg,...
    peadpt_all(:,1),peadpt_all(:,2),pe_adpt_avg,...
    te_all(:,1),te_all(:,2),te_non_avg,teadpt_all(:,1),teadpt_all(:,2),te_adpt_avg];
xlswrite(filename,write,sheet,'A2');

sheetsum = ['summary page beta',num2str(beta)];
write = {' ';'PE baseline';'PE non-adpt';'PE adpt';'TE non-adpt';'TE adpt';...
    'p';'TE non MC_avg 1';'TE non MC_avg 2';'TE adpt MC_avg 1';'TE adpt MC_avg 2'};
xlswrite(filename,write,sheetsum,'A1');
xlswrite(filename,{type},sheetsum,[char(65+idx),'1']);
xlswrite(filename,{[num2str(pe_base_avgMC,'%.2f'),'$\pm$',num2str(sem_pe_base,'%.2f')]},sheetsum,[char(65+idx),'2']);
xlswrite(filename,{[num2str(pe_non_avgMC,'%.2f'),'$\pm$',num2str(sem_pe_non,'%.2f')]},sheetsum,[char(65+idx),'3']);
xlswrite(filename,{[num2str(pe_adpt_avgMC,'%.2f'),'$\pm$',num2str(sem_pe_adpt,'%.2f')]},sheetsum,[char(65+idx),'4']);
xlswrite(filename,{[num2str(te_non_avgMC,'%.2f'),'$\pm$',num2str(sem_te_non,'%.2f')]},sheetsum,[char(65+idx),'5']);
xlswrite(filename,{[num2str(te_adpt_avgMC,'%.2f'),'$\pm$',num2str(sem_te_adpt,'%.2f')]},sheetsum,[char(65+idx),'6']);
xlswrite(filename,p,sheetsum,[char(65+idx),'7']);
xlswrite(filename,te_non2',sheetsum,[char(65+idx),'8']);
xlswrite(filename,te_adpt2',sheetsum,[char(65+idx),'10']);

%% plot

figure();
set(0,'defaultfigureposition',[10 10 800 600]);

for ii = 1:n
    D_show(:,ii) = D_all(1,:,ii);
    plot(D_show(:,ii));hold on;
end
for ii = 1:n
    plot(A_non_avg(ii)*ones(Tr,1),'--r');
end
plot([1:Tr],adiag(1:Tr,:),'k',[Tr+1:N],adiag(Tr+1:N,:),':k');
legend('Adptive Estimation','Adptive Estimation','Non-adptive Estimation','Non-adptive Estimation','True','True','Location','southeast')
axis([1 N 0 1.2])
xlabel('time steps');ylabel('eiganvalue');
set(gca,'FontSize',14);

if(Ts < 100)
    show_num = Ts;
else
    show_num = 100;
end

figure()
for ii = 1:l
    subplot(2,l,2*ii-1)
    plot(y_test(ii,1:show_num-1));hold on; plot(yp(ii,2:show_num))
    legend('True','Estimated','Location','southwest')
    xlabel('time steps');ylabel('magnitude');
    axis([1 show_num min(min(y_test(ii,1:show_num-1))) max(max(y_test(ii,1:show_num-1)))])
    title(['Non-adaptive algorithm on output ' num2str(ii)])
    set(gca,'FontSize',14);
    subplot(2,l,2*ii);set(gca,'FontSize',22);
    plot(y_test(ii,1:show_num-1));hold on; plot(ypyy(ii,2:show_num))
    legend('True','Estimated','Location','southwest')
    xlabel('time steps');ylabel('magnitude');
    axis([1 show_num min(min(y_test(ii,1:show_num-1))) max(max(y_test(ii,1:show_num-1)))])
    title(['Adaptive algorithm on output ' num2str(ii)])
    set(gca,'FontSize',14);
end
end