# GaussianConditionalRandomFields

This repository contains MATLAB implementations of the Gaussian Conditional Random Fields (GCRF) model proposed in the following papers:


1. Gligorijevic, Dj., Stojanovic, J., Obradovic, Z. (2016) " Uncertainty Propagation in Long-term Structured Regression on Evolving Networks," Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence (AAAI-16), Phoenix, AZ, February 2016 

2. Stojanovic, J., Jovanovic, M., Gligorijevic, Dj., Obradovic, Z. (2015) " Semi-supervised learning for structured regression on partially observed attributed graphs," Proceedings of the 2015 SIAM International Conference on Data Mining (SDM 2015) Vancouver, Canada, April 30 - May 02, 2015 

3. Gligorijevic, Dj., Stojanovic, J., Obradovic, Z. (2015) " Improving Confidence while Predicting Trends in Temporal Disease Networks," Proceedings of the 4th Workshop on Data Mining for Medicine and Healthcare, 2015 SIAM International Conference on Data Mining, Vancouver, Canada, April 30 - May 02, 2015 


Please cite the relevant paper in the abovementioned list if you use the code in any form.



## Running the Code

The train-test implementation for precipitation prediction is given for the uncertainty propagation capable model (Gligorijevic et. al. 2016) in iterative or direct approach.

You can train and evaluate model(s) by running following lines in the MATLAB:

```
  ./mainCRFPrecipitationIterativeCRC.m
  ./mainCRFPrecipitationDirectCRC.m
```
