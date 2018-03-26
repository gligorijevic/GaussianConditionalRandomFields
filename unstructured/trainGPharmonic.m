

for i = trainigTS

    ylab=y(:,i);
    ylab=y(~isnan(ylab));
    [fu, fu_CMN] = harmonic_function(similarity,ylab );
    y(isnan(y(:,i)),i)=fu;
end