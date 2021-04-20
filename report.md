# Spotify audio features

The aim of this project is to apply probabilistic reasoning to a Bayesian Network created using audio features of songs on Spotify. Audio features are attributes which Spotify automatically calculates for each song in order to describe its mood and character.

In addition to Spotify audio features, the presence of each song in the Billboard Hot 100 Chart [1] is also taken into consideration.

Even though it is not the focus of the present work, using the Billboard data in association with the Spotify features can be particularly interesting, since it can give an insight about the characteristics which are more relevant in making a song a commercial success. For an in-depth analysis, a work by Elena Georgieva, Marcella Suta, and Nicholas Burton [2, 3] is suggested.

## Features

- Loudness: volume in dB
- Energy: perceived energy
- Danceability: regular and loud beat
- Acousticness: presence of acoustic instruments
- Valence: perceived cheerfulness
- Speechiness: amount of spoken words
- Instrumentalness: absence of voice
- Liveness: whether the song is performed live
- Tempo: beats per minute
- Chart: whether the song ended up in the Billboard Hot 100 Chart
- Artist score: whether the artist has had a previous hit

All features are expressed with a value ranging from 0 to 1, with the exception of loudness and tempo.

The data of both the Spotify features and the Billboard ones are taken from the .csv file included in the work by Elena Georgieva, Marcella Suta, and Nicholas Burton [2, 3].

## Bayesian network

Since not all variables are completely independent from each other, the following Bayesian network is used to represent the dependencies. The structure of the diagram is created using a mixture of correlation analysis and human intuition in evaluating the causal links.

![The Bayesian network](images/network.png)

## Code

To start, the .csv file is imported as a pandas `DataFrame`. Then, in order to deal with the continuity of most of the variables, the data are discretized using the pandas `cut` function.

Next, a pgmpy `BayesianModel` is created specifying all the causal links shown in the network and it is then fit to the dataframe data using the `BayesianModel.fit` function.

### Conditional probability distributions

A conditional proability distriutions (CPD) can be calculated using the `BayesianModel.get_cpds` function: as an example, the Danceability CPD is shown.

| Speechiness     | 0     | 1     | 2     | 3     |
| --------------- | ----- | ----- | ----- | ----- |
| Danceability(0) | 0.052 | 0.018 | 0.063 | 0.000 |
| Danceability(1) | 0.264 | 0.113 | 0.125 | 0.304 |
| Danceability(2) | 0.508 | 0.411 | 0.688 | 0.522 |
| Danceability(3) | 0.176 | 0.458 | 0.125 | 0.174 |

### Markov blankets

Markov blankets can be computed through the `BayesianModel.get_markov_blanket` function: as an example, the Danceability Markov blanket returns the list `['ArtistScore', 'Energy', 'Chart', 'Speechiness', 'Valence', 'Loudness']`, whose correctness can be shown by again looking at the network.

### Inferences

Three different kinds of inference are tested and compared. The code used for each inference method can be found in the `functions.py` file.

The first inference method is exact inference, using in particular the variable elimination method, which consists in "do[ing] the calculation once and sav[ing] the results for later use" [4].

The second method is the rejection sampling method, an approximate inference algorithm which "generates samples from the prior distribution specified by the network. Then, it rejects all those that do not match the evidence." [4].

The last algorithm is another approximate inference method, likelihood weighted sampling. This algortithm "avoids the inefficiency of rejection sampling by generating only events that are consistent with the evidence" [4].

The CPDs of P(Valence | Chart=1) computed using the three methods are:

|            | Exact | Rejection | Weighted |
| ---------- | ----- | --------- | -------- |
| Valence(0) | 0.143 | 0.150     | 0.150    |
| Valence(1) | 0.274 | 0.267     | 0.268    |
| Valence(2) | 0.334 | 0.337     | 0.336    |
| Valence(3) | 0.249 | 0.246     | 0.247    |

### Graphs

Since rejection sampling and likelihood weighted sampling are numerical computations, their accuracy depends on the number of samples which are calculated. To compare their performance with the exact values, two different graphs are plotted. In both cases the x axis represents the number of samples in logarithmic scale, while the y axes of the two graphs show respectively the absolute probability and the difference from the exact value. It is clear from the plots that as the sampling size increases, the approximate estimation converge to the reference exact value, as expected.

![Probabilities comparison](images/probabilities.png)

![Differences comparison](images/differences.png)

## Bibliography

[1] Billboard Media. 2021. The Hot 100 Chart. (April 2001). Retrieved April 20, 2021 from https://www.billboard.com/charts/hot-100

[2] Elena Georgieva, Marcella Suta, and Nicholas Burton. 2018. HitPredict: Predicting Hit Songs Using Spotify Data. Retrieved April 20, 2021 from http://cs229.stanford.edu/proj2018/report/16.pdf

[3] Elena Georgieva, Marcella Suta, and Nicholas Burton. 2018. HitPredict: Using Spotify Data to Predict Billboard Hits. Retrieved April 20, 2021 from https://ccrma.stanford.edu/~egeorgie/HitPredict/ICML2020.pdf

[4] Stuart J. Russell and Peter Norvig. 2010. *Artificial Intelligence: A Modern Approach* (3rd. ed.). PearsonEducation, Upper Saddle River, New Jersey.