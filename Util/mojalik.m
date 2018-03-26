function lik = mojalik(alpha, beta)
%ylabel,mu
ylabel=[0.761333284749569];
ualpha=ln(alpha)
ubeta=ln(beta)
R=[1.32473149693383];
[Q1 Q2 b] = calcPrecision(ualpha, ubeta, Data,'training');
Qcelo = 2*(Q1 + Q2);
label=Data.label(1:nt);
% calculate precision matrix for labeled and hidden data in the training set
Qll=Qcelo(label,label);
Qlh=Qcelo(label,~label);
Qhl=Qcelo(~label,label);
Qhh=Qcelo(~label,~label);
% fQll=full(Qll(1:5,1:5))
% fQlh=full(Qlh(1:5,1:5))
% fQhl=full(Qhl(1:5,1:5))
% fQhh=full(Qhh(1:5,1:5))
%Qll,Qlh,Qhl,Qhh  %Qsl=[Qll Qlh; Qhl Qhh];

bl = b(label);
ylabel = Data.y(label);
% bh = b(~label);   %yhidden = Data.y(~label);

% calculate likelihood for training data
%RR_l= chol(Qll);
% mu_l = Qll\bl;
% mu_h=Qhh\bh;
% mu=[mu_l' mu_h']';
% MUL=mu_l;
mu = Qll\bl;
%RR_h= chol(Qhh);

Q=inv(Qll-Qlh/Qhh*Qhl);
% fQll=full(QL(1:5,1:5))
% QL=inv(Qll-Qlh*inv(Qhh)*Qhl);
RR=chol(Q);

% f = calcLikelihood(RR,Qll,ylabel,mu_l);
f = calcLikelihood(RR,Q,ylabel,mu);