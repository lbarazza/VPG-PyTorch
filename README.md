# VPG-PyTorch
Minimalistic implementation of Vanilla Policy Gradient with PyTorch

This is a simple implementation of the Vanilla Policy Gradient (VPG) approach for tackling the reinforcement learning problem. Policy Gradient methods are part of a broader class of methods called policy-based methods. These methods are currently used in many state-of-the-art algorithms as an alternative to value-based methods such as Q-learning.

Policy-Based Methods provide some advantages from value-based methods such as:
- simplicity: they directly optimize for the optimal policy rather than passing through a value or Q function.
- stochasticity: sometimes the optimal policy is a stochastic one (eg. Rock-Paper-Scissors), in such cases value based methods won't work because they can only provide deterministic policies, but Policy-Based methods most likely will.
- continous action spaces: both Policy and Value Based methods work with continous state spaces, but only the first ones work with continous action spaces.

## Vanilla Policy Gradient
Vanilla Policy Gradient works

![alt text](https://spinningup.openai.com/en/latest/_images/math/47a7bd5139a29bc2d2dc85cef12bba4b07b1e831.svg "Policy Gradient Algorithm")
