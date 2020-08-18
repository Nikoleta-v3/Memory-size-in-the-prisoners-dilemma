Data collection for the Moran process.

Running `main.py` will generate (or append to) `main.csv` with calculations
for the Moran process (with N=4) with an optimal memory 1 player that updates
at every generation.

To run:

```
$ python main.py
```

The data set is of the form:

```
Seed, Opponent p1, Opponent p2, Opponent p3, Opponent p4, K, Best response p1, Best response p2, Best response p3, Best respose p4, A_11, A_12, A_21, A_22, x_k, non_dynamic_x_k
```

Approach:

The fixation probability \\(x_i\\) is given by:

\\[
    x_k = \frac{
            1 + \sum_{j=1}^{j-1}\prod_{i=1}^j\gamma_i
                }{
            1 + \sum_{j=1}^{N-1}\prod_{i=1}^j\gamma_i
                }
\\]

where:

\\[
    \gamma_i = \frac{
                 p_{k, k - 1}
                    }{
                 p_{k, k + 1}
                    }
\\]

Where \\(k\\) is the number of best response players in the population, so that
\\(N - k\\) is the number of opponents.

Note that for every value of \\(k\\) there is a different best response player.

In the case of the best response player the transition probabilities depend on
the payoff matrix \\(A ^ {(k)}\\) where:

- \\(A ^ {(k)}\_{11}\\) is the utility of the best response player against
  itself.
- \\(A ^ {(k)}\_{12}\\) is the utility of the best response player against
  the opponent.
- \\(A ^ {(k)}\_{11}\\) is the utility of the opponent against
  the best response player.
- \\(A ^ {(k)}\_{11}\\) is the utility of the opponent against itself.

The matrix \\(A ^ {(k)}\\) is calculated for each value of \\(k\\) once the best
response dynamics algorithm has calculated the best response player.

Using this we can write down the total utilities/fitnesses for each player:

\\[f_1^{(k)} = (k - 1) A_{11}^{(k)} + (N - k)A_{12}^{(k)}\\]

\\[f_2^{(k)} = (k) A_{21}^{(k)} + (N - k - 1)A_{22}^{(k)}\\]

(\\(f_1^{(k)}\\) is the fitness of the best response player, and \\(f_2^{(k)}\\)
is the fitness of the opponent)

Using this we have:

\\[
    p_{k, k - 1} = \frac{
                     (N - k)f_2^{(k)}
                    }{
                     kf_1^{(k)}+(N - k)f_2^{(k)}
                    }
                    \frac{
                     k
                    }{
                     N
                    }
\\]

and:

\\[
    p_{k, k + 1} = \frac{
                     kf_1^{(k)}
                    }{
                     kf_1^{(k)}+(N - k)f_2^{(k)}
                    }
                    \frac{
                     (N - k)
                    }{
                     N
                    }
\\]
