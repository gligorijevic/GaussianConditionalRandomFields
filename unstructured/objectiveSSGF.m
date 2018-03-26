function [ lp, u] = objectiveSSGF( u, x, y, k )
'aaaaaaaaaaaaaaaaaaaaa'
uu_index = isnan(y);
ss_index = ~isnan(y);

W =getW(x,k); %having 1/k if xj is among k nearest neighbors of xi, otherwise 0
A = W + W' - W'*W;
% issing(A)
% size(A)
% sum(diag(A))
I = eye(size(A));

L = I - A;

% issing(L);
M = L + u(2)*I;

% Muu = M(uu_index,uu_index);
% Mus = M(uu_index, ss_index);
% Mss = M(ss_index,ss_index);
% 
% yu = -inv(Muu)*Mus*y(ss_index);

C = (1/u(1))*inv(M);
Cuu = C(uu_index,uu_index);
Cus = C(uu_index, ss_index);
Css = C(ss_index,ss_index);
iCss = inv(Css);

ns = sum(ss_index);

% RR= chol(Css);
% sum(log(diag(RR)))
lp = -0.5 * ((log(det(Css))) - ns*log(u(1)) + u(1)*y(ss_index)'*iCss*y(ss_index))

% beta analitically
beta = ns/(y(ss_index)'*iCss*y(ss_index));

%dericative alpha
C2 = C'*C;
C2 = C2(ss_index,ss_index);
alpha = u(2) + 0.5*sum(diag(iCss*C2)) - (ns/2)*((y(ss_index)'*iCss*C2*iCss*y(ss_index))/(y(ss_index)'*iCss*y(ss_index)));

u = [beta, alpha]

end