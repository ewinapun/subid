% plotting the optimal Ts length
function [meanTs,varTs] = testlenfun(y,u,i,Tr,k,nmax,nx,start,incr,last)
    for Ts = start:incr:last
        [A,B,C,D,K,R,ss,pea] = traintestfun(y,u,i,Tr,Ts,k,nmax,nx);
        meanTs(Ts/incr,:) = mean(pea);
        varTs(Ts/incr,:) = var(pea);
    end
    figure
    subplot(211);plot([start:incr:last],meanTs);title('mean vs Ts');
    subplot(212);plot([start:incr:last],varTs);title('var vs Ts'); 
end