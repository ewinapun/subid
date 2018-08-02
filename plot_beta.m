% plot optimal beta
%PE
auto = fliplr([29.39 20.0520   19.6960   21.5940   27.8310   57.3880]);
step = fliplr([20.96 14.1000   13.3320   19.2780   25.8720   58.2590]);
linear = fliplr([19.82 14.9370   15.8660   18.2150   23.8760   57.0190]);
beta = [0.8 0.95 0.98 0.99 0.995 1];
figure()
plot(beta,auto,':','linewidth',2);hold on
plot(beta,step,'--','linewidth',2);
plot(beta,linear,'linewidth',2);
legend('RAND-WALK','STEP','LINEAR');
xlabel('\beta');ylabel('Prediction Error (%)');
xticks(beta);
set(gca,'FontSize',14);
% set(0,'defaultfigureposition',[10 10 1200 1000]);

%TE
autote = fliplr([10.33  3.6100    4.1600    4.2600   10.4000   25.7500]);
stepte = fliplr([18.69  6.1400    4.5800    5.9900   16.9500   35.5800]);
linearte = fliplr([5.66 1.9100    2.0800    2.3200    3.2600    9.6300]);
figure()
plot(beta,autote,':','linewidth',2);hold on
plot(beta,stepte,'--','linewidth',2);
plot(beta,linearte,'linewidth',2);
legend('RAND-WALK','STEP','LINEAR');
xlabel('\beta');ylabel('Tracking Error (%)');
xticks(beta);
set(gca,'FontSize',14);
% set(0,'defaultfigureposition',[10 10 1200 1000]);