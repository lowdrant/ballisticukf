# ballisticukf
WIP. Estimating axis of rotation using world-frame observations of rigid body points.

## Quickstart
```
> python3 main.py
```
Should work out of the box if you've done some scientific python. Only requires your standard scientific computing packages -- matplotlib, numpy, scipy.

## Motivation
How could we estimate the center-of-mass (CoM) of a rotating rigid body undergoing ballistic motion? Since ballistic objects rotate about their CoM (neglecting air resistance), it should be straightforward to estimate the CoM from world-frame position measurements of points on the body. However, I have yet to find an explicit treatment of the problem. This is my attempt to demonstrate it and do some extra UKF/PF work.

## Treatment
For simplicity, I'm going to work with a spinning disk in moving in 2D.

### High-Level Procedure
1. Generate rigid body transforms from CoM to points on body
1. Simulate disk motion, tracking CoM and angle
1. Compute point motion _after simulation_ using RBTs and disk motion (simpler code)
1. Feed point motion time series into UKF
1. profit

### A Note on Units
Our disk has mass `m` and radius `r` and experiences gravitational acceleration `g`. We can choose units of mass, length, and time so that `m = r = g = 1`. Then all results are dependent on _ratios_, primarily initial velocity to `g`. That way I can focus on actually working without worrying about confounding details like accidentally making mass negative.

## References
[1] Thrun, Sebastian, et al. Probabilistic Robotics. MIT Press, 2010.
 
## Author
Marion Anderson - [lmanderson42@gmail.com](mailto:lmanderson42@gmail.com)
