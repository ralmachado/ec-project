# ec-project

## Project statement
*Mutation Operators for Reals - Evolutionary Computation*

### Brief Description

The most common mutation operator used with reals is the one that changes a gene using a gaussian distribution.
We may proceed differently, like choosing a new value value within the range of possible values based on a uniform distribution, or using a non-uniform distribution as defined below (the gene $x_k$ will be mutated into $x_k'$):

$$
x_k' =
\left\{
\begin{matrix}
x_k + \Delta(t, S_k-x_k) & \text{if }choice(0,1) = 0 \\
x_k - \Delta(t, S_k-x_k) & \text{if }choice(0,1) = 1
\end{matrix}
\right.
$$

where $S_k$ is the upper limit and $I_k$ is the lower limit for $x_k$, and $\Delta(t,y)$ a function that gives you, non uniformly, a result in the interval $[0, y]$ and $t$ is a generation counter.
$\Delta(t,y)$ is defined by:

$$
\Delta(t,y) = y \times r \times \left(1 - \frac{t}{T}\right)^b
$$

where $r$ is a random number between 0 and 1, $T$ the maximum number of generations, and $b$ a system parameter that controls the degree of non uniformity.
Let's see in what way these approaches may lead to different results.

### Goals

Implement two versions of a standard evolutionary algorithm, one for each of the two mutation operators mentioned above.
For each version of the algorithm, and each benchmark problem, do **thirty runs**.
Store quality measures, like the performance at the end of the run.
Do a statistical analysis of those measures and draw your conclusions.

## Notes to report

When using Random Keys with TSP, the domain doesn't affect the results, since the RNG is seeded the same, the resulting permutation is not affected, because the only difference is the resulting scale