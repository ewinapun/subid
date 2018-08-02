clc; clear; close all;

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
%  x_{k+1) = A x_k + B u_k + w_k (= K e_k)       
%    y_k   = C x_k + D u_k + v_k (= e_k)
% cov(v_k) = R
% where v_k is l dimension, w_k is n dimension, and R is lxl dimension

m = 1; l = 1;
N = 40*4000; % length of inputs
a = [0.603 0.603 0 0;-0.603 0.603 0 0;0 0 -0.603 -0.603;0 0 0.603 -0.603];
%a = diag([0.95, 0.9, 0.8, 0.7]); 
b = [1.1650;0.6268;0.751;0.3516];
c = [0.2641 -1.4462 1.2460 0.5774];
d = [-1.3493];
u = randn(N,m);
% [u,t] = gensig('sin',20,(N-1)*0.1,0.1);
y = dlsim(a,b,c,d,u);

% % implementing own dlsim(a,b,c,d,u)
%         x(:,1)=zeros(length(a),1);
%         for k=1:N
%             x(:,k+1) = a*x(:,k)+b*u(k,m);
%             ydlsim(k,m) = c*x(:,k)+d*u(k,m);
%         end

k = [0.1242;-0.0828;0.0390;-0.0225];
r = [0.05];   % A positive definite matrix is a symmetric matrix with all positive eigenvalues
                % square of any symmetric (to get +ve eiganvalue)
e = randn(N,1)*chol(r);
y = dlsim(a,b,c,d,u) + dlsim(a,k,c,eye(1),e);

% subplot(311);plot(u(:,m));title('Input');
% subplot(312);plot(y(:,l));title('Output without noise');
% subplot(313);plot(y(:,1));title('Output with noise');

%% Running subspace id & obtaining prediction errors in different scenario
nmax = 6; % testing up to nmax
i = 2*(nmax)/1; % number of outputs is 1
% 1) SISO Nx known
nx = 4;
%     a) vary n the number of sets
if (0)
    Tr = 1000; Ts = 1000; k = 30;
    [A,B,C,D,K,R,ss,pea] = traintestfun(y,u,i,Tr,Ts,k,nmax,nx);
    figure
    h = histogram(pea);
    h.BinEdges = [0:0.25:10];
    title(['Distribution of prediction error with ',num2str(k), ' sets']);

    for k = 1:30
        [A,B,C,D,K,R,ss,pea] = traintestfun(y,u,i,Tr,Ts,k,nmax,nx);
        meank(k,:) = mean(pea);
        vark(k,:) = var(pea);
    end
    figure
    subplot(211);plot(meank);title('mean vs k');
    subplot(212);plot(vark);title('var vs k');
    % seems like n = 10 is sufficent for a stable mean and var   
%     disp('press any key to continue');
%     pause;
end
%     b) vary Tr
if (0)
    Tr = 100; Ts = 1000; k = 10;
    [A,B,C,D,K,R,ss,pea] = traintestfun(y,u,i,Tr,Ts,k,nmax,nx);
    figure
    h = histogram(pea);
    h.BinEdges = [0:0.25:10];
%     h(1).FaceColor = [1 0 0];h(2).Color = [1 0 0];
    title(['Distribution of prediction errors varying Tr, constant Ts = ',...
           num2str(Ts),' and k = ',num2str(k)]);
    Tr = 1000;
    [A,B,C,D,K,R,ss,pea] = traintestfun(y,u,i,Tr,Ts,k,nmax,nx);
    hold on
    h = histogram(pea);
    h.BinEdges = [0:0.25:10];
%     h(1).FaceColor = [0 1 0];h(2).Color = [0 1 0];
    Tr = 10000;
    [A,B,C,D,K,R,ss,pea] = traintestfun(y,u,i,Tr,Ts,k,nmax,nx);
    hold on
    h = histogram(pea);
    h.BinEdges = [0:0.25:10];
%     h(1).FaceColor = [0 0 1];h(2).Color = [0 0 1];
    legend('Tr = 100','Tr = 1000','Tr = 10000')

    % plotting the optimal Tr length
    [meanTr,varTr] = trainlenfun(y,u,i,Ts,k,nmax,nx,100,100,10000);
    % seems like Tr = 1000 is sufficent for a stable mean
%     disp('press any key to continue');
%     pause;
end
%     c) vary Ts
if (0)
    % plotting the histogram
    Tr = 1000; Ts = 100; k = 10;
    [A,B,C,D,K,R,ss,pea] = traintestfun(y,u,i,Tr,Ts,k,nmax,nx);
    figure
    h = histogram(pea);
    h.BinEdges = [0:0.25:10];
    title(['Distribution of prediction errors varying Ts, constant Tr = ',...
           num2str(Tr),' and k = ',num2str(k)]);
    Ts = 1000;
    [A,B,C,D,K,R,ss,pea] = traintestfun(y,u,i,Tr,Ts,k,nmax,nx);
    hold on
    h = histogram(pea);
    h.BinEdges = [0:0.25:10];
    Ts = 10000;
    [A,B,C,D,K,R,ss,pea] = traintestfun(y,u,i,Tr,Ts,k,nmax,nx);
    hold on
    h = histogram(pea);
    h.BinEdges = [0:0.25:10];
    legend('Ts = 100','Ts = 1000','Ts = 10000')
    
    % plotting the optimal Ts length
    [meanTs,varTs] = testlenfun(y,u,i,Tr,k,nmax,nx,100,100,10000);
    [meanTsshort,varTsshort] = testlenfun(y,u,i,Tr,k,nmax,nx,10,10,1000);
        % zoom in to just Ts = 100 to 1000
    figure
    subplot(211);plot([100:10:1000],meanTsshort(10:100));title('mean vs Ts');
    subplot(212);plot([100:10:1000],varTsshort(10:100));title('var vs Ts');
    % seems like Ts = 1000 is sufficent for a stable variance

end

%% 2) SISO unknown Nx (order of the system)
%     a) mean pe of no testing
if(1)
    disp('(a) Calculating peBar');
    nmax = 20; i = 2*(nmax)/1; 
    Tr = 2000; Ts = 0; k = 20;
    for n = 1:nmax
        disp(['Identifying order n = ',int2str(n)]);
        [A,B,C,D,K,R,ss,pea] = traintestfun(y,u,i,Tr,Ts,k,nmax,n);
        peBar(n,:) = mean(pea);
    end
    figure
    plot([1:nmax],peBar);
%     title('peBar vs system order');
    xlabel('System order');
    axis([0,nmax+1,0,max(peBar)+1]);
    hold on
    
%     b)mean of peHat with testing sets
    disp('(b) Calculating peHat');
    Tr = 2000; Ts = 2000; k = 20;
    for n = 1:nmax
        disp(['Identifying order n = ',int2str(n)]);
        [A,B,C,D,K,R,ss,pea] = traintestfun(y,u,i,Tr,Ts,k,nmax,n);
        peHat(n,:) = mean(pea);
    end
    
    plot([1:nmax],peHat);
%     title('peHat vs system order');
    xlabel('System order');
    axis([0,nmax+1,0,max(peHat)+1]);
    legend('(a)','(b)')
end

%% 3) cross validation
if(0)
    N = 20000; Ts = 1000;
    ytrunc = y(1:N);utrunc = u(1:N);
    for m = 0:N/Ts-1
        if(m == 0)
            y_train = ytrunc((m+1)*Ts+1:end); 
            u_train = utrunc((m+1)*Ts+1:end); 
        elseif(m == N/Ts-1)
            y_train = ytrunc(1:m*Ts); 
            u_train = utrunc(1:m*Ts); 
        else
            y_train = ytrunc([1:m*Ts,(m+1)*Ts+1:end]); 
            u_train = utrunc([1:m*Ts,(m+1)*Ts+1:end]); 
        end
        y_test = ytrunc(m*Ts+1:(m+1)*Ts);
        u_test = utrunc(m*Ts+1:(m+1)*Ts);
        [A,B,C,D,K,R,ss] = Ewina_subid(y_train,u_train,i,nx,[],1);
        [yp,pe] = predic(y_test,u_test,A,B,C,D,K);
        pea(m+1,:) = pe; % accumulation of all pe
    end
    figure
    bar([1:m+1],pea);title('peHat vs system order');
    xlabel('location of Ts (towards beginning <--> towards end)');
    axis([0,m+2,0,max(pea)+1]);
end