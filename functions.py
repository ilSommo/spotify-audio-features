__version__ = '1.0.0-alpha.2'
__author__ = 'Martino Pulici'

import matplotlib.pyplot as plt
import numpy as np
from pgmpy.factors.discrete import State, TabularCPD


def probability_text(variable, evidence={}):
    """Returns the probability text for printing.

    Parameters
    ----------
    variable : str
        The query variable.
    evidence : dict, default {}
        Dictionary of evidence for conditional probability.

    Returns
    -------
    text : str
        Text to print.
    """
    if not evidence:
        text = "P(" + variable + ")"
    else:
        text = "P(" + variable + " |"
        for e in evidence.items():
            text += " " + str(e[0]) + "=" + str(e[1])
        text += ")"
    return(text)


def exact(exact_inference, variable, evidence={}):
    """Prints the exact inference conditional probability distribution.

    Parameters
    ----------
    exact_inference : pgmpy.inference.ExactInference.VariableElimination
    variable : str
        The query variable.
    evidence : dict, default {}
        Dictionary of evidence for conditional probability.
    """
    text = probability_text(variable, evidence)
    text += " – Exact inference"
    query = exact_inference.query([variable], evidence, show_progress=False)
    state_names = query.state_names
    cardinality = query.cardinality[0]
    probabilities = query.values
    cpd = TabularCPD(variable, cardinality, np.reshape(
        probabilities, (cardinality, 1)), state_names=state_names)
    print(text)
    print(cpd)


def likelihood_weighted(
        approximate_inference,
        variable,
        evidence={},
        size=10000):
    """Prints the likelihood weighted sampling conditional probability distribution.

    Parameters
    ----------
    approximate_inference : pgmpy.sampling.Sampling.BayesianModelSampling
    variable : str
        The query variable.
    evidence : dict, default {}
        Dictionary of evidence for conditional probability.
    size : int, default 10000
        Number of size.
    """
    text = probability_text(variable, evidence)
    text += " – Likelihood weighted sampling (" + str(size) + " size)"
    evidence_state = []
    for e in evidence.items():
        evidence_state.append(State(e[0], e[1]))
    query = approximate_inference.likelihood_weighted_sample(
        evidence=evidence_state, size=size)
    state_names = sorted(query[variable].unique())
    cardinality = len(state_names)
    probabilities = []
    for state in state_names:
        probabilities.append([np.sum(
            np.dot(query[variable] == state, query['_weight'])) / np.sum(query['_weight'])])
    cpd = TabularCPD(
        variable,
        cardinality,
        probabilities,
        state_names={
            variable: state_names})
    print(text)
    print(cpd)


def rejection(approximate_inference, variable, evidence={}, size=10000):
    """Prints the rejection sampling conditional probability distribution.

    Parameters
    ----------
    approximate_inference : pgmpy.sampling.Sampling.BayesianModelSampling
    variable : str
        The query variable.
    evidence : dict, default {}
        Dictionary of evidence for conditional probability.
    size : int, default 10000
        Number of size.
    """
    text = probability_text(variable, evidence)
    text += " – Rejection sampling (" + str(size) + " size)"
    evidence_state = []
    for e in evidence.items():
        evidence_state.append(State(e[0], e[1]))
    query = approximate_inference.rejection_sample(
        evidence=evidence_state, size=size, show_progress=False)
    state_names = sorted(query[variable].unique())
    cardinality = len(state_names)
    probabilities = []
    for state in state_names:
        probabilities.append(
            [np.count_nonzero(query[variable] == state) / size])
    cpd = TabularCPD(
        variable,
        cardinality,
        probabilities,
        state_names={
            variable: state_names})
    print(text)
    print(cpd)


def graph_points(
        exact_inference,
        approximate_inference,
        variable,
        evidence,
        starting_size=10,
        final_size=100000,
        experiments=20):
    """Creates the graph points for method comparison.

    Parameters
    ----------
    exact_inference : pgmpy.inference.ExactInference.VariableElimination
    approximate_inference : pgmpy.sampling.Sampling.BayesianModelSampling
    variable : str
        The query variable.
    evidence : dict, default {}
        Dictionary of evidence for conditional probability.
    starting_size : int, default 10
        Starting number of size.
    final_size : int, default 100000
        Final number of size.
    experiments : int, default 20
        Number of experiments

    Returns
    -------
    sizes : list
        List of x coordinates.
    exact_results : dict
        Dictionary of lists of exact inference y coordinates.
    likelihood_weighted_results : dict
        Dictionary of lists of likelihood weighted sampling y coordinates.
    rejection_results : dict
        Dictionary of lists of rejection sampling y coordinates.
    """
    sizes = list(
        np.logspace(
            np.log10(starting_size),
            np.log10(final_size),
            num=experiments,
            dtype='<i8'))
    exact_query = exact_inference.query(
        [variable], evidence, show_progress=False)
    likelihood_weighted_results = {}
    rejection_results = {}
    state_names = exact_query.state_names[variable]
    evidence_state = []
    for e in evidence.items():
        evidence_state.append(State(e[0], e[1]))
    for state in state_names:
        likelihood_weighted_results[state] = []
        rejection_results[state] = []
    for size in sizes:
        for state in state_names:
            likelihood_weighted_query = approximate_inference.likelihood_weighted_sample(
                evidence=evidence_state, size=size)
            likelihood_weighted_probability = np.sum(
                np.dot(
                    likelihood_weighted_query[variable] == state,
                    likelihood_weighted_query['_weight'])) / np.sum(
                likelihood_weighted_query['_weight'])
            likelihood_weighted_results[state].append(
                likelihood_weighted_probability)
            rejection_query = approximate_inference.rejection_sample(
                evidence=evidence_state, size=size, show_progress=False)
            rejection_probability = np.count_nonzero(
                rejection_query[variable] == state) / size
            rejection_results[state].append(rejection_probability)
    exact_results = dict(zip(state_names, exact_query.values))
    return sizes, exact_results, likelihood_weighted_results, rejection_results


def graph(
        sizes,
        exact_results,
        likelihood_weighted_results,
        rejection_results,
        variable,
        evidence):
    """Shows a comparison graph of various inference methods.

    Parameters
    ----------
    sizes : list
        List of x coordinates.
    exact_results : dict
        Dictionary of lists of exact inference y coordinates.
    likelihood_weighted_results : dict
        Dictionary of lists of likelihood weighted sampling y coordinates.
    rejection_results : dict
        Dictionary of lists of rejection sampling y coordinates.
    variable : str
        The query variable.
    evidence : dict, default {}
        Dictionary of evidence for conditional probability.
    """
    text = probability_text(variable, evidence)
    text += " – Probabilities comparison"
    print(text)
    plt.figure(figsize=(20, 5))
    i = 0
    for state in exact_results.keys():
        i += 1
        plt.subplot(1, len(exact_results.keys()), i)
        plt.xlim(sizes[0], sizes[-1])
        plt.ylim(0, 1)
        plt.title(state)
        likelihood_weighted_plot, = plt.semilogx(
            sizes, likelihood_weighted_results[state], label="Likelihood weighted sampling")
        rejection_plot, = plt.semilogx(
            sizes, rejection_results[state], label="Rejection sampling")
        exact_plot, = plt.semilogx(
            sizes, exact_results[state] * np.ones(len(sizes)), label="Exact inference")
        plt.legend(
            handles=[
                exact_plot,
                likelihood_weighted_plot,
                rejection_plot])
    plt.show()


def diff_graph(
        sizes,
        exact_results,
        likelihood_weighted_results,
        rejection_results,
        variable,
        evidence):
    """Shows a comparison graph of the differences between various inference methods.

    Parameters
    ----------
    sizes : list
        List of x coordinates.
    exact_results : dict
        Dictionary of lists of exact inference y coordinates.
    likelihood_weighted_results : dict
        Dictionary of lists of likelihood weighted sampling y coordinates.
    rejection_results : dict
        Dictionary of lists of rejection sampling y coordinates.
    variable : str
        The query variable.
    evidence : dict, default {}
        Dictionary of evidence for conditional probability.
    """
    text = probability_text(variable, evidence)
    text += " – Differences comparison"
    print(text)
    plt.figure(figsize=(20, 5))
    i = 0
    for state in exact_results.keys():
        i += 1
        plt.subplot(1, len(exact_results.keys()), i)
        plt.xlim(sizes[0], sizes[-1])
        plt.ylim(-0.5, 0.5)
        plt.title(state)
        likelihood_weighted_plot, = plt.semilogx(
            sizes, likelihood_weighted_results[state] - exact_results[state], label="Likelihood weighted sampling")
        rejection_plot, = plt.semilogx(
            sizes, rejection_results[state] - exact_results[state], label="Rejection sampling")
        exact_plot, = plt.semilogx(
            sizes, np.zeros(
                len(sizes)), label="Exact inference")
        plt.legend(
            handles=[
                exact_plot,
                likelihood_weighted_plot,
                rejection_plot])
    plt.show()
