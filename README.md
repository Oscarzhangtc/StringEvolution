# String Evolution
A simulation that imitates the biological evolutionary features of [Evolutionary Algorithms](https://en.wikipedia.org/wiki/Evolutionary_algorithm) to evolve a populations of messages(set of strings) to the goal message of your choice.

# Intallation and Usage
Packages DEAP and numpy are used in this simulation.

```bash
$ pip install deap
$ pip install numpy
```
Make sure to install both before running.

Then run:

```bash
$ python evolve_text.py [goal_message]
```

Where goal_message is an optional argument specfying the target text of the evolution.

Scroll to see a sample result!


# Overview

In general, genetic algorithms follow these basic steps:

1. Create an “genotype” encoding of problem solutions
2. Generate an initial population of individuals with random genotypes
3. Using a problem-specific “fitness function”, select the best individuals from the population and kill off the rest
4. Mate the selected individuals to create a new generation of individuals
5. Randomly mutate the genotypes of some individuals
6. Repeat 3-5 for as many generations as required

##### Sample result:

```bash
[Generation 0]
Message('FQJIONGQKONKLADLAFJAX FLFR') [Distance: 34]
Message('AWFEWYQWHJQ') [Distance: 33]
Message('QEFOIQJHOIQJIHLAAWY QWEYQJQ FQIOEPQIAJP') [Distance: 35]
...
[Generation 10]
Message('FQWEFQHJ OFEWIQEFQ AFYULUI') [Distance: 24]
Message('AWE WFEOIAJIJFOI') [Distance: 67]
...
[Generation 20]
Message('AWEFAX WAEFAX PEALLC') [Distance: 17]
...
[Generation 100]
Message('AJFFHIS THIWAOE FI FTAEIOG')  [Distance: 11]
...
[Generation 200]
Message('THIOS JIOS IFANLS WFWEH TJIOARJIOET') [Distance: 12]
...
[Generation 500]
Message('THIS IS THE FINAL TARGET') [Distance: 0]
```


**Step 1-2 Initialization**

The genotype in our case would pieces of texts or sets of strings. To start(step 2), pieces of texts is randomly generate as the initial population of "genotypes".


**Step 3 Levenshtein distance**

The distance value above as seen in sample result is an implementation of the [Levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance), which calculates the distance between the current message and the target text. The algorithm aims to minimize the distance and serves as the fitness function, where genotypes with a large distance are eliminated and smaller distances are favored.

This feature corresponds to the `levenshtein_distance()` function in `evolve_text.py`.


**Step 4 Mating**

Mating is behavior when pieces of the genotypes of two parent Messages combines/mate to produce offspring. To implement this effect, the two point crossover technique is used. The technique will produce following behavior:

```python
  >>> parent1 = "ABCDEF"
  >>> parent2 = "UVWXYZ"
  >>> print(TwoPointCrossover(parent1, parent2))
  ("ABWXYF", "UVCDEZ")
```

This crossover is intended to model homologous recombination of chromosomes that occurs during sexual reproduction. In this case, 2  strings are mated to produce offspring that inherits characters from its parents.

This feature corresponds to the `two_point_crossover()` function in `evolve_text.py`.


**Step 5 Mutation**

Mutation modifies the genotype of a single individual, and is intended to model chromosome mutation. The types of mutation possible in this simulation are:

1. Insertion of a single Message character
2. Deletion of a single Message character
3. Substitution of a single Message character for another random character

This feature corresponds to the `mutate_text()` function in `evolve_text.py`.
