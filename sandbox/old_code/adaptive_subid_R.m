% subspace identification Algorithm written by Ewina
% ds_flag = 2 is for stochastic
% ds_flag = 1 is for deterministic

function [sys] = adaptive_subid_R(R,l,m,i,n,W,ds_flag)

sil = 1;
if (ds_flag == 2);m = 0; end
R = R(1:2*i*(m+l),1:2*i*(m+l)); 	% Truncate size

% Give W its default value
if (nargin < 6);W = [];end
if isempty(W)
  if (ds_flag == 1); W = 'SV'; 		% Deterministic: default to SV
  else;            W = 'CVA';end 	% Stochastic: default to CVA
end


% Check the weight to be used
Wn = 0;
if (length(W) == 2) 
  if (all(W == 'SV') | all(W == 'sv') | all(W == 'Sv'));
    Wn = 1; 
    if (ds_flag == 1);Waux = 2;else;Waux = 3;end
  end
end    
if (length(W) == 3) 
  if (prod(W == 'CVA') | prod(W == 'cva') | prod(W == 'Cva'));
    Wn = 2;
    if (ds_flag == 1);Waux = 3;else;Waux = 1;end
  end 
end
if (Wn == 0);error('W should be SV or CVA');end
W = Wn;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                              BEGIN ALGORITHM
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% **************************************
% STEP 1 - oblique and orthogonal projection
% **************************************

  mi2  = 2*m*i;
  Rf = R((2*m+l)*i+1:2*(m+l)*i,:); 	% Future outputs
  Rp = [R(1:m*i,:);R(2*m*i+1:(2*m+l)*i,:)]; % Past (inputs and) outputs
  if (ds_flag == 1)
    Ru  = R(m*i+1:2*m*i,1:mi2); 	% Future inputs
    % Perpendicular Future outputs 
    Rfp = [Rf(:,1:mi2) - (Rf(:,1:mi2)/Ru)*Ru,Rf(:,mi2+1:2*(m+l)*i)]; 
    % Perpendicular Past
    Rpp = [Rp(:,1:mi2) - (Rp(:,1:mi2)/Ru)*Ru,Rp(:,mi2+1:2*(m+l)*i)]; 
  end
% A/_B C = A/Bpend * pinv(C/Bpend) * C
% obl/Ufp = Yf/Ufp * pinv(Wp/Ufp) * (Wp/Ufp)
% project the future output to the space spanned by the past input and
% output along the direction of past and future input
% The extra projection on Ufp (Uf perpendicular) tends to give better

% Funny rank check (SVD takes too long)
% This check is needed to avoid rank deficiency warnings
  if (ds_flag == 1)
    % Funny rank check (SVD takes too long)
    % This check is needed to avoid rank deficiency warnings
    if (norm(Rpp(:,(2*m+l)*i-2*l:(2*m+l)*i),'fro')) < 1e-10
      Ob  = (Rfp*pinv(Rpp')')*Rp; 	% Oblique projection
    else
      Ob = (Rfp/Rpp)*Rp;
    end
  else    
    % Ob  = (Rf/Rp)*Rp; which is the same as 
    Ob = [Rf(:,1:l*i),zeros(l*i,l*i)];
  end
% **************************************
% STEP 2 calculate SVD of weighted oblique projection
% **************************************

mydisp(sil,'Computing SVD');
% Compute the matrix WOW we want to take an SVD of
% W = 1 (SV), W = 2 (CVA)
% Extra projection of Ob on Uf perpendicular
  if (ds_flag == 1)
    % Extra projection of Ob on Uf perpendicular
    WOW = [Ob(:,1:mi2) - (Ob(:,1:mi2)/Ru)*Ru,Ob(:,mi2+1:2*(m+l)*i)];
  else
    WOW = Ob;
  end    
  if (W == 2)
    W1i = triu(qr(Rf'));
    W1i = W1i(1:l*i,1:l*i)';
    WOW = W1i\WOW;
  end
[U,S,V] = svd(WOW);
ss = diag(S);
clear V S WOW
  
% **************************************
% STEP 3 - determine order and obtain U1 and S1
% **************************************

% if isempty(n)
%     figure(gcf);hold off;subplot;
%     H = bar([1:l*i],ss); xx = get(H,'XData'); yy = get(H,'YData'); 
%     semilogy(xx,yy+10^(floor(log10(min(ss))))); % log scale on y-axis
%     axis([0,length(ss)+1,10^(floor(log10(min(ss)))),10^(ceil(log10(max(ss))))]);
%     title('Singular Values');
%     xlabel('Order');
%     n = 0;
%     while (n < 1) | (n > l*i-1)
%         n = input('System order ? ');
%         if isempty(n);n = -1;end
%     end
% end

U1 = U(:,1:n); 				% Determine U1
S1 = ss(1:n);

% **************************************
% STEP 4 - determine gamma ( observability subspace)
% **************************************

% weighted value of u determined by the ss
% Determine gam and gamm
gam  = U1*diag(sqrt(S1));
gamm = gam(1:l*(i-1),:);
% The pseudo inverses
gam_inv  = pinv(gam);
gamm_inv = pinv(gamm);

% **************************************
% STEP 5 - solve for A and C
% **************************************

% Determine the matrices A and C
mydisp(sil,['Computing System matrices A,C (Order ',num2str(n),')']); 
Rhs = [  gam_inv*R((2*m+l)*i+1:2*(m+l)*i,1:(2*m+l)*i),zeros(n,l) ; ...
    R(m*i+1:2*m*i,1:(2*m+l)*i+l)];
Lhs = [        gamm_inv*R((2*m+l)*i+l+1:2*(m+l)*i,1:(2*m+l)*i+l) ; ...
    R((2*m+l)*i+1:(2*m+l)*i+l,1:(2*m+l)*i+l)];


% Solve least square
sol = Lhs/Rhs;

% Extract the system matrices A and C
A = sol(1:n,1:n);
C = sol(n+1:n+l,1:n);
res = Lhs - sol*Rhs; 			% Residuals

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%   Recompute gamma from A and C
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

gam=C;
for k=2:i
	gam((k-1)*l+1:k*l,:) = gam((k-2)*l+1:(k-1)*l,:)*A;
end
gamm = gam(1:l*(i-1),:);      
gam_inv = pinv(gam);
gamm_inv = pinv(gamm);	

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%   Recompute the states with the new gamma
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Rhs = [  gam_inv*R((2*m+l)*i+1:2*(m+l)*i,1:(2*m+l)*i),zeros(n,l) ; ...
    R(m*i+1:2*m*i,1:(2*m+l)*i+l)];
Lhs = [        gamm_inv*R((2*m+l)*i+l+1:2*(m+l)*i,1:(2*m+l)*i+l) ; ...
    R((2*m+l)*i+1:(2*m+l)*i+l,1:(2*m+l)*i+l)];


% **************************************
% STEP 6 - solve for B and D
% **************************************
if (ds_flag == 2)
  B = [];
  D = [];
else
  mydisp(sil,['Computing System matrices B,D (Order ',num2str(n),')']); 
  % P and Q as on page 125
  P = Lhs - [A;C]*Rhs(1:n,:);
  P = P(:,1:2*m*i);
  Q = R(m*i+1:2*m*i,1:2*m*i); 		% Future inputs

  % L1, L2, M as on page 119
  L1 = A * gam_inv;
  L2 = C * gam_inv;
  M  = [zeros(n,l),gamm_inv];
  X  = [eye(l),zeros(l,n);zeros(l*(i-1),l),gamm];
  
  totm=0;
  for k=1:i
    % Calculate N and the Kronecker products (page 126)
    N = [...
	    [M(:,(k-1)*l+1:l*i)-L1(:,(k-1)*l+1:l*i),zeros(n,(k-1)*l)]
	[-L2(:,(k-1)*l+1:l*i),zeros(l,(k-1)*l)]];
    if k == 1;
      N(n+1:n+l,1:l) = eye(l) + N(n+1:n+l,1:l);
    end
    N = N*X;
    totm = totm + kron(Q((k-1)*m+1:k*m,:)',N);
  end
  
  % Solve Least Squares
  P = P(:);
  sol = totm\P;
  
  % Find B and D
  sol_bd = reshape(sol,(n+l),m);
  D = sol_bd(1:l,:);
  B = sol_bd(l+1:l+n,:);
end

% **************************************
% STEP 7 - determine Q, S, R
% **************************************

if (norm(res) > 1e-10)
  % Determine QSR from the residuals
  mydisp(sil,['Computing System matrices G,L0 (Order ',num2str(n),')']); 
  % Determine the residuals
  cov = res*res'; 			% Covariance
  Qs = cov(1:n,1:n);Ss = cov(1:n,n+1:n+l);Rs = cov(n+1:n+l,n+1:n+l); 
  
  sig = dlyap(A,Qs); % discrete-time Lyapunov equation AXAT ? X + Q = 0,
  G = A*sig*C' + Ss;
  L0 = C*sig*C' + Rs;

  % Determine K and Ro
  mydisp(sil,'Computing Riccati solution')
  [K,Ro] = gl2kr(A,G,C,L0);
else
  Ro = [];
  K = [];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                              END ALGORITHM
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Gathering matrices
sys.A = A;sys.B = B;sys.C = C;sys.D = D;
sys.Ro = Ro;sys.ss = ss;sys.K = K;sys.R = R;

end
