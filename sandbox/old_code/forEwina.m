clear all; clc;

%% Parameters
%Generate data without noise with a diagonal A matrix evolving over time
%        x(k+1)=A(k)x(k)
%        y(k)=Cx(k)
%        A(k)=[1-q1(k),0;0,1-q2(k)]
%        C=I
%
K=2000;
nx=2;
ny=2;
n=2;
horizon=5;
r=zeros(1,K);
C=eye(n,n);
A=cell(K,1);
%% Using one of the followimh 2 methods to generate time variant A

% %I:Generating time variant diagonal elements of matrix A with exponential decay. 

% for k=1:K
%     %A{k}=[0.7,0;0,0.95];
%     %A{k}=[0.1*exp(-k/200)+0.8,0;0,0.1*exp(-k/200)+0.85];% System ID is Ok with beta =0.6.
%     A{k}=[0.05*exp(-k/1000)+0.95,0;0,0.05*exp(-k/1000)+0.90];% 
%     %A{k}=[0.2*exp(-k/200)+0.7,0;0,0.2*exp(-k/200)+0.8];%
%     %A{k}=[0.05*exp(-k/1000)+0.95,0;0,0.05*exp(-k/1000)+0.95];% Ok (is captured properly with beta=0.6)
%     %A{k}=[0.05*exp(-k/1000)+0.93,0;0,0.04*exp(-k/1000)+0.95];
% end
% 
% %Ploting the real poles of A matrix over time
% figure
% hold on
% for k=1:K
% plot(k,A{k}(1,1),'*')
% 
% end
% figure
% hold on
% for k=1:K
% plot(k,A{k}(2,2),'*')
% end
% %%
% %II: I:Generating time variant diagonal elements of matrix A with random walk.
% 
% A{1}=zeros(2,2);
% for k=2:K
%     A{k}=0.9998*A{k-1}+[randn,0;0,randn]*0.001;
% end
% for k=1:K
%     A{k}(1,1)=A{k}(1,1)+0.95;
%     A{k}(2,2)=A{k}(2,2)+0.9;
% end
% 
% 
% diag_element_A=zeros(2,K);
% for k=1:K
%     diag_element_A(1,k)=A{k}(1,1);
%     diag_element_A(2,k)=A{k}(2,2);
% %plot(k,A{k}(1,1)+0.95)%,'.b')
% end
% figure
% subplot(2,1,1)
% plot(diag_element_A(1,:));
% subplot(2,1,2);
% plot(diag_element_A(2,:));
% %%
%III: I:Generating time variant diagonal elements of matrix A with sinus wave.


for k=1:K
% f=0.0001;    
%     A{k}(1,1)=0.025*sin(2*pi*f*k)+0.975;
%     A{k}(2,2)=0.025*sin(2*pi*f*k)+0.955;
%     f=0.0001;
%     A{k}(1,1)=0.05*sin(2*pi*f*k)+0.95;
%     A{k}(2,2)=0.05*sin(2*pi*f*k)+0.95;
 A{k}(1,2)=0;
 A{k}(2,1)=0;
 A{k}(1,1)=0.95;
 A{k}(2,2)=0.5;
% A{k}(1,2)=0;
% A{k}(2,1)=0;
end

% t=1:K;
% figure
% subplot(2,1,1)
% plot(0.025*sin(2*pi*f*t)+0.975);
% subplot(2,1,2)
% plot(0.025*sin(2*pi*f*t)+0.925);

% t=1:K;
% figure
% subplot(2,1,1)
% plot(0.05*sin(2*pi*f*t)+0.95);
% subplot(2,1,2)
% plot(0.05*sin(2*pi*f*t)+0.95);

%%
M=2;
D1_for_average=nan(M,K-2*horizon);
D2_for_average=nan(M,K-2*horizon);
T1_for_average=nan(M,K-2*horizon);
T2_for_average=nan(M,K-2*horizon);
T3_for_average=nan(M,K-2*horizon);
T4_for_average=nan(M,K-2*horizon);
%A_hat_for_average=nan(M,K,n,n);
parfor m=1:M
x=zeros(n,K);y=zeros(n,K);
x(:,1)=[1000;1500];
%x(:,1)=[0,0];
for k=1:K-1
    
    x(:,k+1)=A{k}*x(:,k)+randn(nx,1)*30;
    y(:,k)=C*x(:,k);%+randn(ny,1)*0.1;
end
figure;
hold on
plot(x(1,:));
plot(x(2,:));

%% Subspace System Identification (Yuxiao's method)

%% Order Estimation
% step=1;
% maxorder=2;
% errortol=0.1;
% %data_train=y;
%[ nx, aic_values, relative_aic, ordertosearch ] = order_estimation_lin(data_train', maxorder, step, errortol ); 
    %% initialize the QR decompozition

beta= 1;% 0.9995; 
data_train=y;
%temp=y(:,1:2*horizon);
[ R_initial ] = qr_initialization(data_train, horizon, beta ); 
    %% Adaptive estimation of system matrices

L = size(data_train,2)-(2*horizon); 
R_past=R_initial;

% c_estimated=zeros(L,ny,nx);
% a_estimated=zeros(L,nx,nx);
% q_estimated=zeros(L,nx,nx);
% r_estimated=zeros(L,nx,ny);
%%
sys_adpt_overtime=[];
% yhat=zeros(ny,L);
% xhat=x(:,1);% Is it a correcft initialization for adaptive system?
for i=1:L
[ sys_adpt, R_new ] = adaptive_train_onesteponly( data_train,i,R_past, nx, ny, horizon,beta ); 
R_past=R_new;
% Are the following 2 steps (prediction steps of estimation)  correct for the adaptive case?
sys_adpt_overtime=[sys_adpt_overtime;sys_adpt];
%xhat = sys_adpt.a * xhat + sys_adpt.k * (data_train(:,i) - sys_adpt.c*xhat);
%yhat(:, i) = (sys_adpt.c*xhat)';

% a_estimated(i,:,:)=sys_adpt.a;
% c_estimated(i,:,:)=sys_adpt.c;
% q_estimated(i,:,:)=sys_adpt.q;
% r_estimated(i,:,:)=sys_adpt.r;
end
%%
% figure;
% subplot(2,1,1)
% %ylim([-5000,5000]);
% hold on;
% plot(yhat(1,:),'b');
% plot(1:L,y(1,1:L),'r');
% 
% 
% 
% 
% subplot(2,1,2)
% %ylim([-5000,5000]);
% hold on;
% plot(yhat(2,:),'b');
% plot(1:L,y(2,1:L),'r');


%[V,D]=eig(sys_adpt.a);
%% Diagonalize estimated A and compare Tranformations (T) across time
V=zeros(L,n,n);
D=zeros(L,n,n);
D_sorted=zeros(L,2);
%A_hat_trialm=nan(L,n,n);
for k=1:L
    [V(k,:,:),D(k,:,:)]=eig(sys_adpt_overtime(k).a);
    D_sorted(k,:)=sort(diag(abs(squeeze(D(k,:,:)))));
    %A_hat_trialm(k,:,:)=sys_adpt_overtime(k).a;
end
%A_hat_average=mean(A_hat_for_average
%%
%% Ploting eigen vector elements over time
% figure
% subplot(nx^2,1,1);
% plot(1:L,V(:,1,1));
% subplot(nx^2,1,2);
% plot(1:L,V(:,2,1));
% subplot(nx^2,1,3);
% plot(1:L,V(:,1,2));
% subplot(nx^2,1,4);
% plot(1:L,V(:,2,2));

%% ploting eigen values of estimated versus real A over time
% figure
% 
% subplot(nx,1,1);
% ylim([0.8,1.1]);
% hold on;
% plot(1:L,D(:,1,1));
% for k=1:K
% plot(k,A{k}(1,1),'.r')
% end
% 
% subplot(nx,1,2);
% ylim([0.8,1.1]);
% hold on;
% 
% plot(1:L,D(:,2,2));
% for k=1:K
% plot(k,A{k}(2,2),'.r')
% end 
 D1_for_average(m,:)=D_sorted(:,1);
 D2_for_average(m,:)=D_sorted(:,2);
 T1_for_average(m,:)=V(:,1,1);
 T2_for_average(m,:)=V(:,2,1);
 T3_for_average(m,:)=V(:,1,2);
 T4_for_average(m,:)=V(:,2,2);
 %A_hat_for_average(m,:,:,:)=A_hat_trialm;
 
% D_for_average(m,2,:)=D(:,1,1);
end
%%
diag_element_A=zeros(2,K);
for k=1:K
    diag_element_A(1,k)=A{k}(1,1);
    diag_element_A(2,k)=A{k}(2,2);
%plot(k,A{k}(1,1)+0.95)%,'.b')
end
%A_hat_mean=mean(A_hat_for_average);
%%
figure
subplot(2,1,1)
hold on
plot(diag_element_A(2,:),'r');
plot(nanmean(abs(D1_for_average)),'b');
ylim([0,1.2]);
subplot(2,1,2);

hold on
plot(diag_element_A(1,:),'r');
plot(nanmean(abs(D2_for_average)),'b');
ylim([0,1.2]);
%%
figure
subplot(4,1,1)
plot(nanmean(abs(T1_for_average)),'b');

subplot(4,1,2)
plot(nanmean(abs(T2_for_average)),'b');
subplot(4,1,3)
plot(nanmean(abs(T3_for_average)),'b');
subplot(4,1,4)
plot(nanmean(abs(T4_for_average)),'b');
