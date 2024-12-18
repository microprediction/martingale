# martingale puzzles 

Nowcasting of martingales observed with noise

### Puzzles:

1. A generative model where martingales play a central role
2. A benchmark heuristic online algorithm
3. An evaluation function to score algorithms against the benchmark

### Algorithm development goals:

- Algorithms should be close to scale-free
- Algorithms should fit in an online fashion, generally processing data points in several milliseconds or less
- Algorithms should adapt quickly to changing regimes

### Generative models and assessment 

The generative model assessment comprises three stages: 

  1. Generation of parameters from a diffuse prior and
  2. Burn-in period to approximate sampling from the ergodic distribution
  3. A weighted running computation of a cost function where early data points are more important that later ones 
  
We'd like to encourage algorithms to learn quickly. 

