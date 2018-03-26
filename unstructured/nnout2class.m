% class2col.m
%
% transforms the output from a neural network to class prediction

function class = nnout2class(nnout,class_names)

[a,b] = max(nnout');
class = class_names(b');

return;