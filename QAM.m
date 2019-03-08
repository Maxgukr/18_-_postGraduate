eb_n0 = 0:0.001:16;
pb4 = qfunc(sqrt(2*10.^(eb_n0/10)));
ps4 = 2*qfunc(sqrt(10.^(eb_n0/10)));

pb8 = (11/24)*qfunc(sqrt(10.^(eb_n0/10)*(6/((3+sqrt(3))))));
PB8 = (11/24)*qfunc(sqrt(10.^(eb_n0/10)*2.19));
ps8 = qfunc(sqrt(10.^(eb_n0/10)*(2/((3+sqrt(3))))));

pb16 = (3/4)*qfunc(sqrt((4/5)*10.^(eb_n0/10)));
ps16 = (4)*qfunc(sqrt((1/5)*10.^(eb_n0/10)));


figure;
semilogy(eb_n0, pb4,'b-',eb_n0,pb8,'g-',eb_n0,pb16,'r-',[0,16],[0.02,0.02],'k:');
grid on
%plot([0,18],[0.02,0.02],'k');
legend('4-QAM','8-QAM','16-QAM');
title('Bit error probability curve for QAM modulation');
xlabel('Eb/N0(dB)');
ylabel('BER');


%
% figure;
% semilogy(eb_n0,pb8,'b-',eb_n0,PB8,'r-')
% grid on
% legend('8-QAM');
% title('Bit error probability curve for QAM modulation');
% xlabel('Es/N0(dB)');
% ylabel('SER');
%
%hold on;
%semilogy(eb_n0, pb8);
%plot(eb_n0,pb8,'g-');

%hold on;
%semilogy(pb16);
%plot(eb_n0,pb16,'r-');
