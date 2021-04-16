import numpy as np
import pandas as pd
from pgmpy.factors.discrete import State, TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianModel
from pgmpy.sampling import BayesianModelSampling

def exact(exact_inference, variable, evidence = {}):
	
	if not evidence:
		text = "P(" + variable + ")"
	else:
		text = "P(" + variable + " |"
		for e in evidence.items():
			text += " " + str(e[0]) + "=" + str(e[1])
		text += ")"
	text += " – Exact inference"

	query = exact_inference.query([variable], evidence, show_progress = False)
	state_names = query.state_names
	cardinality = query.cardinality[0]
	probabilities = query.values
	cpd = TabularCPD(variable, cardinality, np.reshape(probabilities, (cardinality, 1)), state_names = state_names)
  
	print(text)
	print(cpd)

def likelihood_weighted(approximate_inference, variable, evidence = {}, size = 10000):
	
	if not evidence:
		text = "P(" + variable + ")"
	else:
		text = "P(" + variable + " |"
		for e in evidence.items():
			text += " " + str(e[0]) + "=" + str(e[1])
		text += ")"
	text += " – Likelihood weighted sample (" + str(size) + " samples)"
	
	evidence_state = []
	for e in evidence.items():
		evidence_state.append(State(e[0], e[1]))
	query = approximate_inference.likelihood_weighted_sample(evidence = evidence_state, size = size)
	state_names = list(query[variable].unique())
	state_names.sort()
	cardinality = len(state_names)
	probabilities = []
	for value in state_names:
		probabilities.append([np.sum(np.dot(query[variable] == value, query['_weight'])) / np.sum(query['_weight'])])
	cpd = TabularCPD(variable, cardinality, probabilities, state_names = {variable: state_names})
  
	print(text)
	print(cpd)

def rejection(approximate_inference, variable, evidence = {}, size = 10000):
	
	if not evidence:
		text = "P(" + variable + ")"
	else:
		text = "P(" + variable + " |"
		for e in evidence.items():
			text += " " + str(e[0]) + "=" + str(e[1])
		text += ")"
	text += " – Rejection sample (" + str(size) + " samples)"
	
	evidence_state = []
	for e in evidence.items():
		evidence_state.append(State(e[0], e[1]))
	query = approximate_inference.rejection_sample(evidence = evidence_state, size = size, show_progress = False)
	state_names = list(query[variable].unique())
	state_names.sort()
	cardinality = len(state_names)
	probabilities = []
	for value in state_names:
		probabilities.append([np.count_nonzero(query[variable] == value) / size])
	cpd = TabularCPD(variable, cardinality, probabilities, state_names = {variable: state_names})
  
	print(text)
	print(cpd)