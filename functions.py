#TODO formatting
import matplotlib.pyplot as plt
import numpy as np
from pgmpy.factors.discrete import State, TabularCPD

def probability_text(variable, evidence):
	if not evidence:
		text = "P(" + variable + ")"
	else:
		text = "P(" + variable + " |"
		for e in evidence.items():
			text += " " + str(e[0]) + "=" + str(e[1])
		text += ")"
	return(text)

def exact(exact_inference, variable, evidence = {}):
	
	text = probability_text(variable, evidence)
	text += " – Exact inference"

	query = exact_inference.query([variable], evidence, show_progress = False)
	state_names = query.state_names
	cardinality = query.cardinality[0]
	probabilities = query.values
	cpd = TabularCPD(variable, cardinality, np.reshape(probabilities, (cardinality, 1)), state_names = state_names)

	print(text)
	print(cpd)

def likelihood_weighted(approximate_inference, variable, evidence = {}, size = 10000):
	
	text = probability_text(variable, evidence)
	text += " – Likelihood weighted sampling (" + str(size) + " samples)"
	
	evidence_state = []
	for e in evidence.items():
		evidence_state.append(State(e[0], e[1]))
	query = approximate_inference.likelihood_weighted_sample(evidence = evidence_state, size = size)
	state_names = list(query[variable].unique())
	state_names.sort()
	cardinality = len(state_names)
	probabilities = []
	for state in state_names:
		probabilities.append([np.sum(np.dot(query[variable] == state, query['_weight'])) / np.sum(query['_weight'])])
	cpd = TabularCPD(variable, cardinality, probabilities, state_names = {variable: state_names})

	print(text)
	print(cpd)

def rejection(approximate_inference, variable, evidence = {}, size = 10000):
	
	text = probability_text(variable, evidence)
	text += " – Rejection sampling (" + str(size) + " samples)"
	
	evidence_state = []
	for e in evidence.items():
		evidence_state.append(State(e[0], e[1]))
	query = approximate_inference.rejection_sample(evidence = evidence_state, size = size, show_progress = False)
	state_names = list(query[variable].unique())
	state_names.sort()
	cardinality = len(state_names)
	probabilities = []
	for state in state_names:
		probabilities.append([np.count_nonzero(query[variable] == state) / size])
	cpd = TabularCPD(variable, cardinality, probabilities, state_names = {variable: state_names})

	print(text)
	print(cpd)

def graph_points(exact_inference, approximate_inference, variable, evidence, starting_size = 10, final_size = 100000, experiments = 20):
	evidence_state = []
	for e in evidence.items():
		evidence_state.append(State(e[0], e[1]))
	exact_query = exact_inference.query([variable], evidence, show_progress = False)
	likelihood_weighted_results = {}
	rejection_results = {}
	sizes = list(np.logspace(np.log10(starting_size), np.log10(final_size), num = experiments, dtype = '<i8'))
	state_names = exact_query.state_names[variable]
	for state in state_names:
		likelihood_weighted_results[state] = []
		rejection_results[state] = []
	for size in sizes:
		for state in state_names:
			likelihood_weighted_query = approximate_inference.likelihood_weighted_sample(evidence = evidence_state, size = size)
			likelihood_weighted_probability = np.sum(np.dot(likelihood_weighted_query[variable] == state, likelihood_weighted_query['_weight'])) / np.sum(likelihood_weighted_query['_weight'])			
			likelihood_weighted_results[state].append(likelihood_weighted_probability)
			rejection_query = approximate_inference.rejection_sample(evidence = evidence_state, size = size, show_progress = False)
			rejection_probability = np.count_nonzero(rejection_query[variable] == state) / size
			rejection_results[state].append(rejection_probability)
	return sizes, dict(zip(state_names, exact_query.values)), likelihood_weighted_results, rejection_results

def graph(sizes, exact_results, likelihood_weighted_results, rejection_results, variable, evidence):
	text = probability_text(variable, evidence)
	print(text)

	plt.figure(figsize=(20,5))
	i = 1
	for state in exact_results.keys():
		plt.subplot(1, len(exact_results.keys()), i)
		i += 1
		plt.ylim(0,1)
		plt.title(state)
		likelihood_weighted_plot, = plt.semilogx(sizes, likelihood_weighted_results[state],label="Likelihood weighted sampling")
		rejection_plot, = plt.semilogx(sizes, rejection_results[state],label="Rejection sampling")
		exact_plot, = plt.semilogx(sizes, exact_results[state]*np.ones(len(sizes)),label="Exact inference")
		plt.legend(handles=[exact_plot,likelihood_weighted_plot,rejection_plot])
	plt.show()