Stability of defection, optimisation of strategies and the limits of
memory in the Prisoner's Dilemma.
--------------------------------------

**Authors**: @Nikoleta-v3 @drvinceknight

This repository contains the source code for a paper titled 
"Stability of defection, optimisation of strategies and the limits of
memory in the Prisoner's Dilemma".

## Software

A conda environment specifying all versions of libraries used is given in
`environment.yml`. To create and activate this environment run:

```
$ conda env create -f environment.yml
$ source activate opt-mo
```

The source code for this work have been packaged under the name `opt-mo`.
Install the source code by running the following command
(ensure that you have activated the environment):

```
$ python setup.py install
```

The package has been tested with `pytest`. To test run:

```
$ sh test.sh
```

The entire analysis described in the paper can be found in `nbs/`.

The software is released under an MIT license.
