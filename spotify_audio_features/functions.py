__version__ = '1.2.0'
__author__ = 'Martino Pulici'


import matplotlib.pyplot as plt
import numpy as np
from pgmpy.factors.discrete import State, TabularCPD


def diff_graph(
        sizes,
        exact_results,
        rejection_results,
        weighted_results,
        variable,
        evidence):
    """Shows a comparison graph of the differences between various inference methods.

    Parameters
    ----------
    sizes : list
        List of x coordinates.
    exact_results : dict
        Dictionary of lists of exact inference y coordinates.
    rejection_results : dict
        Dictionary of lists of rejection sampling y coordinates.
    weighted_results : dict
        Dictionary of lists of likelihood weighted sampling y coordinates.
    variable : str
        The query variable.
    evidence : dict, default {}
        Dictionary of evidence for conditional probability.
    """
    # Text to print
    text = probability_text(variable, evidence)
    text += " – Differences comparison"
    # Print text
    print(text)

    # Create plot
    plt.figure(figsize=(20, 5))
    # Subplot index
    i = 0
    # Cycle states
    for state in exact_results.keys():
        # Increase index
        i += 1
        # Create subplot
        plt.subplot(1, len(exact_results.keys()), i)
        # Plot parameters
        plt.xlim(sizes[0], sizes[-1])
        plt.ylim(-0.5, 0.5)
        plt.title(state)
        # Plot results
        rejection_plot, = plt.semilogx(
            sizes, rejection_results[state] - exact_results[state], label="Rejection sampling")
        weighted_plot, = plt.semilogx(
            sizes, weighted_results[state] - exact_results[state], label="Likelihood weighted sampling")
        exact_plot, = plt.semilogx(
            sizes, np.zeros(
                len(sizes)), label="Exact inference")
        # Draw legend
        plt.legend(
            handles=[
                exact_plot,
                rejection_plot,
                weighted_plot])
    # Save plot
    plt.savefig('images/differences.png', facecolor='white')
    # Show plot
    plt.show()


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
    # Text to print
    text = probability_text(variable, evidence)
    text += " – Exact inference"

    # Exact inference query
    query = exact_inference.query([variable], evidence, show_progress=False)
    state_names = query.state_names
    cardinality = query.cardinality[0]
    probabilities = query.values
    # CPD to print
    cpd = TabularCPD(variable, cardinality, np.reshape(
        probabilities, (cardinality, 1)), state_names=state_names)

    # Print text and CPD
    print(text)
    print(cpd)


def graph(
        sizes,
        exact_results,
        rejection_results,
        weighted_results,
        variable,
        evidence):
    """Shows a comparison graph of various inference methods.

    Parameters
    ----------
    sizes : list
        List of x coordinates.
    exact_results : dict
        Dictionary of lists of exact inference y coordinates.
    rejection_results : dict
        Dictionary of lists of rejection sampling y coordinates.
    weighted_results : dict
        Dictionary of lists of likelihood weighted sampling y coordinates.
    variable : str
        The query variable.
    evidence : dict, default {}
        Dictionary of evidence for conditional probability.
    """
    # Text to print
    text = probability_text(variable, evidence)
    text += " – Probabilities comparison"
    # Print text
    print(text)

    # Create plot
    plt.figure(figsize=(20, 5))
    # Subplot index
    i = 0
    # Cycle states
    for state in exact_results.keys():
        # Increase index
        i += 1
        # Create subplot
        plt.subplot(1, len(exact_results.keys()), i)
        # Plot parameters
        plt.xlim(sizes[0], sizes[-1])
        plt.ylim(0, 1)
        plt.title(state)
        # Plot results
        rejection_plot, = plt.semilogx(
            sizes, rejection_results[state], label="Rejection sampling")
        weighted_plot, = plt.semilogx(
            sizes, weighted_results[state], label="Likelihood weighted sampling")
        exact_plot, = plt.semilogx(
            sizes, exact_results[state] * np.ones(len(sizes)), label="Exact inference")
        # Draw legend
        plt.legend(
            handles=[
                exact_plot,
                rejection_plot,
                weighted_plot])
    # Save plot
    plt.savefig('images/probabilities.png', facecolor='white')
    # Show plot
    plt.show()


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
    rejection_results : dict
        Dictionary of lists of rejection sampling y coordinates.
    weighted_results : dict
        Dictionary of lists of likelihood weighted sampling y coordinates.
    """
    # List of x coordinates
    sizes = list(
        np.logspace(
            np.log10(starting_size),
            np.log10(final_size),
            num=experiments,
            dtype='<i8'))

    # Exact inference query
    exact_query = exact_inference.query(
        [variable], evidence, show_progress=False)
    # Rejection sampling y coordinates
    rejection_results = {}
    # Likelihood weighted y coordinates
    weighted_results = {}
    # States
    state_names = exact_query.state_names[variable]
    # Evidence
    evidence_state = []
    # Cycle evidence
    for e in evidence.items():
        # Add evidence
        evidence_state.append(State(e[0], e[1]))
    # Cycle states
    for state in state_names:
        # Append empty lists
        rejection_results[state] = []
        weighted_results[state] = []
    # Cycle x coordinate
    for size in sizes:
        # Cycle states
        for state in state_names:
            # Rejection sampling query
            rejection_query = approximate_inference.rejection_sample(
                evidence=evidence_state, size=size, show_progress=False)
            # Rejection sampling probability
            rejection_probability = np.count_nonzero(
                rejection_query[variable] == state) / size
            # Rejection sampling results
            rejection_results[state].append(rejection_probability)
            # Likelihood weighted query
            weighted_query = approximate_inference.likelihood_weighted_sample(
                evidence=evidence_state, size=size)
            # Likelihood weighted probability
            weighted_probability = np.sum(
                np.dot(
                    weighted_query[variable] == state,
                    weighted_query['_weight'])) / np.sum(
                weighted_query['_weight'])
            # Likelihood weighted results
            weighted_results[state].append(
                weighted_probability)
    # Exact inference y coordinates
    exact_results = dict(zip(state_names, exact_query.values))

    return sizes, exact_results, rejection_results, weighted_results


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
    # Enter for non-conditional probabilities
    if not evidence:
        text = "P(" + variable + ")"
    else:
        text = "P(" + variable + " |"
        for e in evidence.items():
            text += " " + str(e[0]) + "=" + str(e[1])
        text += ")"

    return text


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
    # Text to print
    text = probability_text(variable, evidence)
    text += " – Rejection sampling (" + str(size) + " samples)"

    # Evidence
    evidence_state = []
    for e in evidence.items():
        evidence_state.append(State(e[0], e[1]))
    # Approximate inference query
    query = approximate_inference.rejection_sample(
        evidence=evidence_state, size=size, show_progress=False)
    state_names = sorted(query[variable].unique())
    cardinality = len(state_names)
    probabilities = []
    for state in state_names:
        probabilities.append(
            [np.count_nonzero(query[variable] == state) / size])
    # CPD to print
    cpd = TabularCPD(
        variable,
        cardinality,
        probabilities,
        state_names={
            variable: state_names})

    # Print text and CPD
    print(text)
    print(cpd)


def weighted(
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
    # Text to print
    text = probability_text(variable, evidence)
    text += " – Likelihood weighted sampling (" + str(size) + " samples)"

    # Evidence
    evidence_state = []
    for e in evidence.items():
        evidence_state.append(State(e[0], e[1]))
    # Sampling conditional probability query
    query = approximate_inference.likelihood_weighted_sample(
        evidence=evidence_state, size=size)
    state_names = sorted(query[variable].unique())
    cardinality = len(state_names)
    probabilities = []
    for state in state_names:
        probabilities.append([np.sum(
            np.dot(query[variable] == state, query['_weight'])) / np.sum(query['_weight'])])
    # CPD to print
    cpd = TabularCPD(
        variable,
        cardinality,
        probabilities,
        state_names={
            variable: state_names})

    # Print text and CPD
    print(text)
    print(cpd)
