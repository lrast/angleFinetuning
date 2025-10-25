# Studying fine tuning and multitask learning in an angle discrimination task

## Large data, large models:

### Setup:
Uses Resnet-18 backbone with 2-layer probing, and rotated versions of the Celeb-A dataset from torchvision.

Uses robust estimators of neural activity mean and covariance and the Mahalanobis distance to estimate (linear) Fisher information in the neural activity.
