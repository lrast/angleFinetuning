# Studying fine tuning and multitask learning in an angle discrimination task

## Large data, large models:

### Setup:
Used Resnet-18 backbone with 2-layer probing, and rotated versions of the Celeb-A dataset from torchvision.



## Highlights of prior work:
1. Develops multiple ways to measure Fisher information neural networks
    - We observe very 'rough' Fisher information measurements, in the sense that the can change quickly with small changes in inputs

2. Fisher information iterations show increasing randomness, due to diffusion, without compensating adaptation on small scales. 
    - There are also strong effects of the pre-training in the Fisher information, which show up as small-scale correlations between different fine-tune replicates.

3. Models do specialize to regions of the domain. However, they are often able to extrapolate well outside of the concentrated regions, and the scale of the difference between regions is similar to the noise between replicates, and to the variability of the Fisher information regions.
    - This concentration does effect later fine-tuning. The results appear reproducible, but again fairly small compared to the variation between initializations.
    - Potentially effects from loss functions, but this is hard to assess.

4. Many choices about the models have an effect on the observed patterns:
    - Smoother (differentiable) non-linearities result in smoother Fisher information.
    - Decoding method impacts Fisher information distribution: angular encodings result in multiple-peaked Fisher information.
    - Train-set vs test-set images produce different distributions of encoded values, due to generalization.

5. Conditioning on these specific choices, we do see loss-function dependent concentration of Fisher information.
    - Concentration does happen, but the way that it does does not map very cleanly onto the theory.
    - The additional noise from generalization makes this much less clean for test than for training data.

6. Iterative approaches remain troublesome in terms of convergence.
