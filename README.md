# Overview
This repository contains a fork and extension of an Active Appearance Model (AAM). The goal of this version is to provide a more complete, flexible, and perceptually accurate AAM pipeline by adding missing functions to the code and enhancing the modeling stage.
The original implementation focused primarily on grayscale image synthesis and feature extraction, which means missing part of input during process. It lacked several essential components required for end-to-end model operation. In this fork, the missing functions have been added, enabling full parameter extraction, shape and texture reconstruction, and model-driven image generation. These additions also allow the model to accept landmark coordinates in the more practical and common format of a Numpy 3D array (number of samples, number of landmarks, 2), where the final dimension represents x–y positions. This style of input improves usability and compatibility.
Two major improvements distinguish this version from the original:
## 1.	Full RGB Appearance Modeling
The original code extracted model parameters from color images but converted textures to grayscale during reconstruction, limiting visual fidelity and reducing the expressive power of the appearance model.
This fork extends the AAM to support true RGB texture modeling and synthesis, preserving color information throughout the pipeline. As a result, generated images more closely match the input distribution and better reflect variations in skin tone, hair color, and other critical visual features.
## 2.	Illumination (Light) Normalization
Variations in lighting conditions can significantly affect texture representation and model fitting. To improve robustness, this fork introduces a light normalization stage based on iterative per-image linear normalization (z-score style). In this case, global brightness (offset) and contrast (scale) are estimated and removed.
This technique reduces illumination-driven variability and leads to more consistent texture modeling and reconstruction across heterogeneous datasets.
### Comparing Face Generation in original with new version
Below two faces have been predicted by both versions to have a better comparision

Face1:
![Face1 Fitted by old version](https://github.com/Sobhanh99/Active-Appearance-Model-AAM-/blob/master/Face1%20Prediction%20with%20original%20version.png)
![Face1 Fitted by New version](https://github.com/Sobhanh99/Active-Appearance-Model-AAM-/blob/master/Face1%20Prediction%20with%20enhanced%20version.png)

Face2:
![Face2 Fitted by old version](https://github.com/Sobhanh99/Active-Appearance-Model-AAM-/blob/master/Face2%20Prediction%20with%20original%20version.png)
![Face2 Fitted by New version](https://github.com/Sobhanh99/Active-Appearance-Model-AAM-/blob/master/Face2%20Prediction%20with%20enhanced%20version.png)

## Conclusion
Overall, this fork offers a more complete and perceptually accurate version of Active Appearance Model. It is well-suited for research applications involving face modeling, reconstruction, and analysis where visual realism and robustness are important.


# References:

## Papers
http://www2.imm.dtu.dk/pubdb/pubs/124-full.html     
https://link.springer.com/content/pdf/10.1023/B:VISI.0000029666.37597.d3.pdf    
https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/cootes-eccv-98.pdf    

## Code
https://github.com/krishnaw14/CS736-assignments     
https://github.com/TimSC/image-piecewise-affine

