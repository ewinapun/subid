% Tr is the length of training set
% Ts is the length of testing set
% k is the number of set
% nmax is the maximum order
% nx is order of the system

function [yp,pea,allsys] = adaptive_traintest(y,u,i,Tr,Ts,k,nmax,nx)

% Check the arguments
if (nargin < 5);error('traintestfun needs at least five arguments');end
if (nargin < 6);k = 1;end
if (nargin < 7);nmax = 10;end
if (nargin < 8);nx = 4;end
if (length(y)<(Tr+Ts)*k);error('Not enough output data for traintestfun');end

for m = 0:k-1
%     disp(['training set #',num2str(m+1)]);
    y_train = y(1+m*(Tr+Ts):Tr+m*(Tr+Ts),:);
    y_test =  y(Tr+1+m*(Tr+Ts):Tr+Ts+m*(Tr+Ts),:);
    u_train = u(1+m*(Tr+Ts):Tr+m*(Tr+Ts),:); 
    u_test = u(Tr+1+m*(Tr+Ts):Tr+Ts+m*(Tr+Ts),:);
    allsys = cell(Tr-2*i+1,1);
    R_last = [];
    for t = 1:Tr-2*i+1
        [as] = adaptive_subid(y_train,u_train,i,nx,[],1,t);
%         [as] = adaptive_subid_R(y_train,u_train,i,nx,[],1,t,R_last);
%         R_last = R;
        allsys{t,1} = as;
        clear as
    end
     [as] = adaptive_subid(y_train,u_train,i,nx,[],1,Tr-2*i+1);
%     [as] = adaptive_subid_R(y_train,u_train,i,nx,[],1,Tr-2*i+1,R_last);
    if(Ts~=0)
        [yp,pe] = predic(y_test,u_test,as.A,as.B,as.C,as.D,as.K);
    else
        [yp,pe] = predic(y_train,u_train,as.A,as.B,as.C,as.D,as.K);
    end
    pea(m+1,:) = pe; % accumulation of all pe of set k
end

end
