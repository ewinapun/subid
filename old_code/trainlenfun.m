% plotting the optimal Tr length
function [meanTr,varTr] = trainlenfun(y,u,i,Ts,k,nmax,nx,start,incr,last)
    for Tr = start:incr:last
        [A,B,C,D,K,R,ss,pea] = traintestfun(y,u,i,Tr,Ts,k,nmax,nx);
        meanTr(Tr/incr,:) = mean(pea);
        varTr(Tr/incr,:) = var(pea);
    end
    figure
    subplot(211);plot([start:incr:last],meanTr);title('mean vs Tr');
    subplot(212);plot([start:incr:last],varTr);title('var vs Tr'); 
end