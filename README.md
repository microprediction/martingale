# martingale puzzles 

### Puzzles:

1. A generative model where martingales play a central role
2. A benchmark heuristic online algorithm
3. An evaluation function to score algorithms against the benchmark

### Algorithm development goals:

- Algorithms should be close to scale-free
- Algorithms should fit in an online fashion, generally processing data points in several milliseconds or less
- Algorithms should adapt quickly to changing regimes

### Generative models and assessment 

The generative model comprises two stages: 
  1. Generation of parameters from a diffuse prior and 
  2. Generation of a stochastic process of some kind (possibly multi-dimensional, possibly structured)
  
Multiple sequences are used in the evaluation simultaneously

