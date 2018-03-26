function [ prodtraceTemp ] = blokTraceInvMatrix( matrix,dMatrix,N,T )
prodtraceTemp=0;
for t=1:T
    Matrixblock=matrix((t-1)*N+1:t*N,(t-1)*N+1:t*N);
    rr = chol(Matrixblock);
    irr=inv(rr);
    invMatrixblock=irr*irr';    
    prodtraceTemp= prodtraceTemp+trace(invMatrixblock*dMatrix((t-1)*N+1:t*N,(t-1)*N+1:t*N));    
    
end

end

