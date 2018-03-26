function [ ] = issing( A )
%Check whether matrix x is singular

if( rcond(full(A)) < 1e-12 )
    disp('Matrix is singular');
else
    disp('Matrix is NOT singular');
end

