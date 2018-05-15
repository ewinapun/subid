close all; clc; clear all;

Tr = 5000; Ts = 500; k = 1; % dvide train-test sets
N = (Tr + Ts) * k; % length of inputs
ainit = [0.6,0.8];
adiag = TV_matrix_A(ainit,Tr,Ts,k,'stepd');
   
%     plotting the TV system of matrix a
figure()
plot([1:Tr],adiag(1:Tr,:),'k',[Tr+1:N],adiag(Tr+1:N,:),':k');    
axis([0,N,0,1]);
xlabel('time steps');ylabel('magnitude');
set(0,'defaultfigureposition',[10 10 800 600]);
% 
% filename = 'test-05-08-2018';
% sheet = ['1'];
% pwrite = {char(177),'2';adiag(:,1),adiag(:,2)};
% xlswrite(filename,pwrite,sheet,'A1');
% pwrite = [adiag(:,1),adiag(:,2)];
% xlswrite(filename,pwrite,sheet,'D1');