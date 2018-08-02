% plot test len
auto = [20.23 20.12 20.34 20.95 24.01];
step = [13.2500   13.6900   13.3200   13.3600   14.5600];
linear = [14.5400   14.6700   15.7600   16.7800   23.0300];
% auto = auto - mean(auto);step = step - mean(step); linear = linear - mean(linear);
len = [10 100 500 1000 5000];
figure()
semilogx(len,auto,':','linewidth',2);hold on;
semilogx(len,step,'--','linewidth',2);
semilogx(len,linear,'linewidth',2);
legend('RAND-WALK','STEP','LINEAR');
xlabel('Test length');ylabel('Prediction Error (%)');
xticks(len);xlim([10 5000]);
set(gca,'FontSize',14);
set(0,'defaultfigureposition',[10 10 1200 1000]);