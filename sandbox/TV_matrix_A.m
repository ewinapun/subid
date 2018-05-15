function [adiag] = TV_matrix_A(ainit,Tr,Ts,k,style)
    [~,n] = size(ainit);
    adiag = []; as = [];
    N = Tr + Ts;
    et = 0.000*randn(N/k,n);
    t = linspace(0,1,N/k)'; 
    mag = 0.19;
    if (length(style) == 3) 
        for j = 1:n
            if(prod(style == 'exp'))            % deterministic
                as(:,j) = ainit(j)+mag*(exp(-5*j*t)-1);
            elseif(prod(style == 'sin'))
                as(:,j) = ainit(j)+mag*sin(8*t);
            elseif(prod(style == 'non'))
                as(:,j) = ainit(j)+0*t;
            elseif(prod(style == 'tri'))
                as(1:N/2/k,j) = ainit(j)+0.2*t(1:N/2/k);
                as(N/2/k+1:N/k,j) = as(N/2/k,j)-mag*(t(N/2/k+1:N/k)-0.5);
            elseif(prod(style == 'lin'))
                as(:,j) = ainit(j)+mag*t;            
            end
        end
    end
    
    if (length(style) == 4)
        
        if(prod(style == 'auto'))              % autoregressive model
            % random walk: end point as start point values            
            as(1,:) = ainit; endpoint = ainit + mag;
            flag = 1;
            lin = [];
            for j = 1:n
                lin = [lin,linspace(ainit(j),endpoint(j),N/k)'];
            end
            while(flag)
                as = 0.005*randn(N/k-1,n);     % change n = 1 if equal eig
                as = as - mean(as);
                as = as + (endpoint - ainit)/(N-1);
                as1 = cumsum([ainit;as]);
%                 figure;plot([1:N],as1);axis([0,N,0,1]);
                pb = 0.001; sb = 0.01; rip = 1; dB = 20;
                d = designfilt('lowpassfir', ...
                    'PassbandFrequency',pb,'StopbandFrequency',sb, ...
                    'PassbandRipple',rip,'StopbandAttenuation',dB, ...
                    'DesignMethod','equiripple');
                as2 = filtfilt(d,as1);
                as = filtfilt(d,as2)+0.1*ones(N/k,n);
%                 [as,~] = eegfilt(as1',0.5,0,0.01);as = as';
                if(max(max(as)) < 1 && min(min(as)) > 0 && mean(as(:,1)<as(:,n))==1)
                    flag = 0;
                end
            end
        end
    end
        
    if (length(style) == 5)
        if(prod(style == 'stepu'))
            for j = 1:n
                as(:,j) = ainit(j)+0*t;
                as(Tr/2/k+1:end,j) = as(Tr/2/k+1:end,j)+mag;
            end
            
        elseif(prod(style == 'stepd'))
            for j = 1:n
                as(:,j) = ainit(j)+0*t;
            end
                as(Tr/2/k+1:end,1) = as(Tr/2/k+1:end,1)-mag/2;
                as(Tr/2/k+1:end,2) = as(Tr/2/k+1:end,2);

        elseif(prod(style == 'stlin'))
            for j = 1:n
                as(:,j) = ainit(j)+2*mag*t;
                as(Tr/2/k+1:end,j) = as(Tr/2/k+1:end,j)-mag;
            end
        end
    end
    
    for i = 1:k
        adiag = [adiag;as+et];
    end

end