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
%  x_{k+1) = A x_k + B u_k + w_k (= K *e(k))       
%    y_k   = C x_k + D u_k + v_k (= e(k))
% cov(v_k) = R
% where v_k is l dimension, w_k is n dimension, and R is lxl dimension

N = 40000; % length of inputs
m = 2;l = 2;n = 4;
a = diag([0.1 0.2 0.7 0.8]);
b = [1.1650,-0.6965;0.6268 1.6961;0.0751,0.0591;0.3516 1.7971];
c = [0.2641,-1.4462,1.2460,0.5774;0.8717,-0.7012,-0.6390,-0.3600];
d = zeros(2,2);

u = randn(N,m);
mrand = 0.05*randn(n+l); mu = (mrand + mrand')/2; cov = mu * mu';
R1 = cov(1:n,1:n); R12 = cov(1:n,n+1:n+l); R2 = cov(n+1:n+l,n+1:n+l); 
% generate the output
x(:,1)=zeros(n,1);
%Solve discrete-time algebraic Riccati equations
% A'PA-E'PE-(A'XB+S)inv(B'PB+R)(B'PA+S')+Q = 0
% [X,L,G] = dare(A,B,Q,R,S,E)
[P,V,G] = dare(a',c',R1,R2,R12,eye(n)); 
Kg = (a*P*c'+R12)*inv(c*P*c'+R2);    % Kalman gain
L = c*P*c'+R2;    % cov of e(t)
e = randn(N,l)*chol(L);
for k=1:N
    x(:,k+1) = a * x(:,k)+b * u(k,:)'+Kg * e(k,:)';
    y(k,:) = (c * x(:,k)+d * u(k,:)'+e(k,:)')';
end

%% Running subspace id & obtaining prediction errors

nmax = 10; % testing up to nmax
i = 2*(nmax)/l; % number of outputs is 1
[A,B,C,D,K,R,ss] = Ewina_subid(y,u,i,n,[],1);
[yp,pe] = predic(y,u,A,B,C,D,K);
pe
% bode for discrete state space system
% dbode(a,b,c,d,Ts,IU,W)
w = [0:0.005:0.2]*(2*pi); 		% Frequency vector
m1 = dbode(a,b,c,d,1,1,w); m2 = dbode(a,b,c,d,1,2,w);
M1 = dbode(A,B,C,D,1,1,w); M2 = dbode(A,B,C,D,1,2,w);
m1diff = m1-M1;m1error = sqrt(sum(m1diff.*m1diff)/length(m1diff))
m2diff = m2-M2;m2error = sqrt(sum(m2diff.*m2diff)/length(m2diff))

% converting state space to transfer function
% ss2tf(a,b,c,d,IU)
[numm1,denm1] = ss2tf(a,b,c,d,1);tfm1 = dbode(numm1,denm1,1,w);
[numm2,denm2] = ss2tf(a,b,c,d,2);tfm2 = dbode(numm2,denm2,1,w);
[numM1,denM1] = ss2tf(A,B,C,D,1);tfM1 = dbode(numM1,denM1,1,w);
[numM2,denM2] = ss2tf(A,B,C,D,2);tfM2 = dbode(numM2,denM2,1,w);
tf1diff = tfm1-tfM1;tf1error = sqrt(sum(tf1diff.*tf1diff)/length(tf1diff))
tf2diff = tfm2-tfM2;tf2error = sqrt(sum(tf2diff.*tf2diff)/length(tf2diff))

figure(1);
hold off;subplot;
subplot(221);set(gca,'FontSize',14);
line(w/(2*pi),m1(:,1),'color','b');
hold on;plot(w/(2*pi),M1(:,1),'r','LineWidth',2);
title('bode of input 1 and output 1');
xlabel('w');ylabel('magnitude');

subplot(223);set(gca,'FontSize',14);
line(w/(2*pi),m2(:,1),'color','b');
hold on;plot(w/(2*pi),M2(:,1),'r','LineWidth',2);
title('bode of input 2 and output 1');
xlabel('w');ylabel('magnitude');

subplot(222);set(gca,'FontSize',14);
line(w/(2*pi),m1(:,2),'color','b');
hold on;plot(w/(2*pi),M1(:,2),'r','LineWidth',2);
title('bode of input 1 and output 2');
xlabel('w');ylabel('magnitude');

subplot(224);set(gca,'FontSize',14);
line(w/(2*pi),m2(:,2),'color','b');
hold on;plot(w/(2*pi),M2(:,2),'r','LineWidth',2);
title('bode of input 2 and output 2');
xlabel('w');ylabel('magnitude');


figure(2);
hold off;subplot;
subplot(221);set(gca,'FontSize',14);
line(w/(2*pi),tfm1(:,1),'color','b');
hold on;plot(w/(2*pi),tfM1(:,1),'r','LineWidth',2);
title('bode of input 1 and output 1');
xlabel('w');ylabel('magnitude');

subplot(223);set(gca,'FontSize',14);
line(w/(2*pi),tfm2(:,1),'color','b');
hold on;plot(w/(2*pi),tfM2(:,1),'r','LineWidth',2);
title('bode of input 2 and output 1');
xlabel('w');ylabel('magnitude');

subplot(222);set(gca,'FontSize',14);
line(w/(2*pi),tfm1(:,2),'color','b');
hold on;plot(w/(2*pi),tfM1(:,2),'r','LineWidth',2);
title('bode of input 1 and output 2');
xlabel('w');ylabel('magnitude');

subplot(224);set(gca,'FontSize',14);
line(w/(2*pi),tfm2(:,2),'color','b');
hold on;plot(w/(2*pi),tfM2(:,2),'r','LineWidth',2);
title('bode of input 2 and output 2');
xlabel('w');ylabel('magnitude');


clear m1 m2 M1 M2
m1 = dbode(a,Kg,c,eye(2),1,1,w); m2 = dbode(a,Kg,c,eye(2),1,2,w);
M1 = dbode(A,K,C,eye(2),1,1,w); M2 = dbode(A,K,C,eye(2),1,2,w);
m1diff = m1-M1;noise1error = sqrt(sum(m1diff.*m1diff)/length(m1diff))
m2diff = m2-M2;noise2error = sqrt(sum(m2diff.*m2diff)/length(m2diff))

figure(3);
hold off;subplot;
subplot(221);set(gca,'FontSize',14);
line(w/(2*pi),m1(:,1),'color','b');
hold on;plot(w/(2*pi),M1(:,1),'r','LineWidth',2);
title('bode of noise 1 and output 1');
xlabel('w');ylabel('magnitude');

subplot(223);set(gca,'FontSize',14);
line(w/(2*pi),m2(:,1),'color','b');
hold on;plot(w/(2*pi),M2(:,1),'r','LineWidth',2);
title('bode of noise 2 and output 1');
xlabel('w');ylabel('magnitude');

subplot(222);set(gca,'FontSize',14);
line(w/(2*pi),m1(:,2),'color','b');
hold on;plot(w/(2*pi),M1(:,2),'r','LineWidth',2);
title('bode of noise 1 and output 2');
xlabel('w');ylabel('magnitude');

subplot(224);set(gca,'FontSize',14);
line(w/(2*pi),m2(:,2),'color','b');
hold on;plot(w/(2*pi),M2(:,2),'r','LineWidth',2);
title('bode of noise 2 and output 2');
xlabel('w');ylabel('magnitude');

% help dbode
% help ss2tf

% figure(2);
% for q = 1:l
%     subplot(2,1,q);set(gca,'FontSize',14);
%     line([0:1:N/10-1],[y(1:N/10,q) yp(1:N/10,q)]);
%     title(['Output ',num2str(q)]);
%     xlabel('time (s)');ylabel('magnitude');
% end
% 
% Tr = 2000; Ts = 2000; k = 10;
% for order = 1:nmax
% %     disp(['Identifying order n = ',int2str(order)]);
%     [A,B,C,D,K,R,ss,pea] = traintestfun(y,u,i,Tr,Ts,k,nmax,order);
%     pe(order,:) = mean(pea);
% end
% figure(3);
% subplot(211);set(gca,'FontSize',14);
% line([1:nmax],pe(:,1),'LineWidth',2);
% title('Prediction Error of Output 1 vs system order');
% xlabel('System order');ylabel('Prediction Error');
% axis([0,nmax+1,0,100]);
% subplot(212);set(gca,'FontSize',14);
% line([1:nmax],pe(:,2),'LineWidth',2);
% title('Prediction Error of Output 2 vs system order');
% xlabel('System order');ylabel('Prediction Error');
% axis([0,nmax+1,0,100]);
