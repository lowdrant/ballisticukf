# ballisticukf
WIP. Estimating axis of rotation using world-frame observations of rigid body points.

## Quickstart
```
> python3 main.py
```
Should work out of the box if you've done some scientific python. Only requires your standard scientific computing packages -- matplotlib, numpy, scipy.

## Motivation
How could we estimate the center-of-mass (CoM) of a rotating rigid body in free-fall (undergoing ballistic motion)? Since ballistic objects rotate about their CoM (neglecting air resistance), it should be straightforward to estimate the CoM from world-frame position measurements of points on the body. However, I have yet to find an explicit treatment of the problem. This is my attempt to demonstrate it and do some extra UKF work.

## Treatment

## Author
Marion Anderson - [lmanderson42@gmail.com](mailto:lmanderson42@gmail.com)
