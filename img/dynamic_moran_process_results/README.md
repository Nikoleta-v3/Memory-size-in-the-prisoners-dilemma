# Dynamic Moran Process Results

A figure showing 3 subfigures:

    - first figure: Violin plot of x_K (4 Violins: 1 for each value of K and 1 overall)
    - second figure: Violin plot of SSE (4 Violins: 1 for each value of K and 1 overall)
    - third figure: Violin plot of the x_K / non_dynamic_x_K (4 "Violins" 1 for each p*i)

## Generate plot

Copy latest version of the data with:

    scp siren:Memory-size-in-the-prisoners-dilemma/moran_data/moran/data.csv .

To generate:

    $ python main.py
