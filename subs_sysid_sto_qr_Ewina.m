
% 
%   Stochastic subspace identification (Algorithm 3 in Overschee book)
%   working directly with QR decompositions
%
%           [sys] = subs_sysid_sto_qr(r_mt, ny, nx, horizon, W);
% 
%   Inputs:
%           r_mtx: r matrix in rq decomposition
%           ny: y dim, nx: x dim, horizon: number of block rows, W:
%           weighting method
%         
%           
%   Outputs:
%          sys: the identified system, a structure with system matrices A,
%          C, Q, R, S, K,  NoiseVariance
%          These matrices are used in two equivalent systems
%           1) used for kalman estimation
%               x(t+1)  = A x(t) + w(t)
%               y(t) = C x(t) +v(t)
%               with E{[w(t) v(t)]'[w(t) v(t)]} = [Q, S;S',R]
%           2) used for prediction
%               x(t+1) = A x(t) + K e(t)
%               y(t) = C x(t) + e(t)
%               with E{e(t)'e(t)} = NoiseVariance
%                
%   Optional:
%
%           [sys, sv] = subs_sysid_sto(y,n,W);
%   
%           n:    optional order estimate (default [])
%           W:    optional weighting flag
%                    CVA: canonical variate analysis (default)
%                    PC:  principal components
%                    UPC: unweighted principal components
%           sv:   column vector with singular values
%           
%   Note:        
%           The computed system is ALWAYS positive real
%
%   Example:
%   
%             [sys, ss] = sto_pos(y,10,'CVA');
%           
%   Reference:
%   
%           Subspace Identification for Linear Systems
%           Theory - Implementation - Applications
%           Peter Van Overschee / Bart De Moor
%           Kluwer Academic Publishers, 1996, Page 90 (Fig 3.13)
%           
%  Written by 
%           Yuxiao Yang, University of Souther California
%           June 4th, 2015
%           Ajusted code from the Overchee and Moor codes 
%           from website http://homes.esat.kuleuven.be/~smc/sysid/software/
%           
%

function [sys] = subs_sysid_sto_qr_Ewina(r_mt, ny, nx, horizon, W, beta);



% Compute the R factor (see Chapter 6 for the qr decomposition)
  l = ny; 
  i = horizon; 
  n = nx; 
  R = r_mt'; 
  R = R(1:2*i*l,1:2*i*l); 		% Truncate
  
  

%if (l < 0);error('Need a non-empty output vector');end
%if ((ny-2*i+1) < (2*l*i));error('Not enough data points');end
Wn = 0;
if (length(W) == 3) 
  if (prod(W == 'CVA') | prod(W == 'cva') | prod(W == 'Cva'));Wn = 1;end 
  if (prod(W == 'UPC') | prod(W == 'upc') | prod(W == 'Upc'));Wn = 3;end
end    
if (length(W) == 2) 
  if (prod(W == 'PC') | prod(W == 'pc') | prod(W == 'Pc'));Wn = 2;end 
end
if (Wn == 0);error('W should be CVA, PC or UPC');end
W = Wn;
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                                  BEGIN ALGORITHM
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% **************************************
%               STEP 1 
% **************************************

% First compute the orthogonal projections Ob 
 % Ob  = R(l*i+1:2*l*i,:);
 
Rf=R(l*i+1:2*l*i,:);
Rp=R(1:l*i,:);
Ob=Rf*Rp'*pinv(Rp*Rp')*Rp;

Rf_p=R(l*(i+1)+1:2*l*i,:);
Rp_p=R(1:l*(i+1),:);
Ob_p=Rf_p*Rp_p'*pinv(Rp_p*Rp_p')*Rp_p;
% **************************************
%               STEP 2 
% **************************************

% Compute the SVD
  % Compute the matrix WOW we want to take an svd off
  % W == 1 (CVA), W == 2 (PC), W == 3 (UPC)
  if (W == 1)
    W1i = triu(qr(R(l*i+1:2*l*i,1:2*l*i)'));
    W1i = W1i(1:l*i,1:l*i)';
    WOW = W1i\Ob;
  end
  if (W == 2)
    WOW = R(l*i+1:2*l*i,1:l*i)*R(1:l*i,1:l*i)';
  end
  if (W == 3)
    WOW = Ob;
  end
  [U,S,V] = svd(WOW);
  if (W == 1);U = W1i*U;end
  sv = diag(S);
  clear V S WOW;

% **************************************
%               STEP 3 
% **************************************

 U1 = U(:,1:n); 				% Determine U1
 
% **************************************
%               STEP 4 
% **************************************

% Determine gam and gamm
gam  = U1*diag(sqrt(sv(1:n)));
gamm = gam(1:l*(i-1),:);
gamup = gam(l+1:end,:);

% And their pseudo inverses
gam_inv  = pinv(gam);
gamm_inv = pinv(gamm);
  
% **************************************
%               STEP 5 
% **************************************

% Determine the states Xi and Xip
Xi  = gam_inv  * Ob;
%Xip = gamm_inv * R(l*(i+1)+1:2*l*i,:);
Xip = gamm_inv *Ob_p;

% % Extract the system matrices
A = gamm_inv * gamup;
C = gam(1:l,:);
% **************************************
%               STEP 2a 
% **************************************

% Determine the state matrices A and C
Rhs = [       Xi   ]; 	% Right hand side
Lhs = [      Xip   ;  R(l*i+1:l*(i+1),:)]; % Left hand side

beta_sum =0;
% % Determine the residuals
res = Lhs - [A;C] * Rhs; 			% Residuals
for row = 1:n+l
%     res(:,row) = beta^(-(n+l-row)/2)*res(:,row);
    beta_sum = beta_sum + beta^(-(n+l-row)/2);
end
cov = res*res'/beta_sum; 			% Covariance
Qs = cov(1:n,1:n);Ss = cov(1:n,n+1:n+l);Rs = cov(n+1:n+l,n+1:n+l);

% **************************************
%               STEP 5a 
% **************************************
% Determine K and NoiseVariance
sig = dlyap(A,Qs);
G = A*sig*C' + Ss;
L0 = C*sig*C' + Rs;
[K,NoiseVariance] = gl2kr(A,G,C,L0);
NoiseVariance = triu(NoiseVariance)+triu(NoiseVariance,1)';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                                  END ALGORITHM
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Gathering matrices
sys.a = A;
sys.c = C;
sys.q = Qs;
sys.r = Rs;
sys.s = Ss;
sys.k = K;
sys.noisevariance = NoiseVariance;

end

