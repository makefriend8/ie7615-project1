| Model | Total parameters | Trainable parameters | Fit time (s) | Inference (ms) | Validation loss | Test correct | Test accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Custom_CNN | 11,963,782 | 11,963,782 | 15.04 | 6.42 | 0.5215 | 17/24 | 70.83% |
| ResNet_Frozen | 23,577,094 | 12,294 | 24.21 | 144.68 | 0.3526 | 21/24 | 87.50% |
| ResNet_Finetuned | 23,577,094 | 23,531,654 | 53.34 | 144.18 | 0.0300 | 23/24 | 95.83% |
