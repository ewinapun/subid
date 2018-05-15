function H = recursive_blkhank(y,i,j,beta)
    % Make a (block)-row vector out of y
    [l,ny] = size(y); if ny < l;y = y';[l,ny] = size(y);end
    
    % Check dimensions
    if i < 0 || j < 0;error('blkHank: i & j should be positive');end
    if j > ny-i+1;disp(ny-i+1);error('blkHank: j too big');end
    
    % Make a block-Hankel matrix
    H=zeros(l*i,j);
    for k=1:i
        H((k-1)*l+1:k*l,:)=y(:,k:k+j-1);
    end
    for q = 1:j
            H(:,q)= H(:,q)*beta.^((j-q)/2);
    end
end

