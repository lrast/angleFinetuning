# Studying fine tuning and multitask learning in an angle discrimination task


## experiment ideas:
1. adapation to different distribution of angles
2. hysteresis in adaptation to wide vs narrow distributions
3. continual learning on different distributions
4. Compare the results of different trainers, with and without saving the trainer state.



## results:
1. Saving the trainer state prevents initialization pollution, while saving only the network weights does not!



## improvements:
1. Experiment with alternative metrics, for example calibration curves


## Thinking about what I need to do now:
I'm at a point where there are several things that I need to accomplish to push to the next phase of the project
1. termination conditions for the iteration: based on the variance of the distirbutions, or maybe the KL divergence or something like that
2. Add mean reversion to the iteration to prevent the growth of noise between iterates. Ideally, we want this to act on a small scale only.




