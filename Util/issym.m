function [ ] = issym( x )
%This function checks whether the input matrix is a symetric or not.

% issym = all(all(tril(x)==triu(x).'));
issym =  all(all(x==x.'));

if issym == 1
    disp('The matrix is symetric!\n');
else
    disp('The matrix is NOT symetric!\n');
end;

end

