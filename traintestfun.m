% Tr is the length of training set
% Ts is the length of testing set
% k is the number of set
% nmax is the maximum order
% nx is order of the system

function [A,B,C,D,K,R,ss,pea] = traintestfun(y,u,i,Tr,Ts,k,nmax,nx)

% Check the arguments
if (nargin < 5);error('traintestfun needs at least five arguments');end
if (nargin < 6);k = 1;end
if (nargin < 7);nmax = 10;end
if (nargin < 8);nx = 4;end
if (length(y)<(Tr+Ts)*k);error('Not enough output data for traintestfun');end

for m = 0:k-1
    y_train = y(1+m*(Tr+Ts):Tr+m*(Tr+Ts),:);
    y_test =  y(Tr+1+m*(Tr+Ts):Tr+Ts+m*(Tr+Ts),:);
    u_train = u(1+m*(Tr+Ts):Tr+m*(Tr+Ts),:); 
    u_test = u(Tr+1+m*(Tr+Ts):Tr+Ts+m*(Tr+Ts),:);
    [A,B,C,D,K,R,ss] = Ewina_subid(y_train,u_train,i,nx,[],1);
    if(Ts~=0)
        [yp,pe] = predic(y_test,u_test,A,B,C,D,K);
    else
        [yp,pe] = predic(y_train,u_train,A,B,C,D,K);
    end
    pea(m+1,:) = pe; % accumulation of all pe of set k
end
end