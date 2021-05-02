import matplotlib.pyplot as plt
from sklearn import tree
import pandas as pd
import os
import graphviz
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz, DecisionTreeRegressor

def graphRF(data_path, label='data'):
	# Load the seaweed dataset:
	df = pd.read_csv(data_path)
	
	print(df.columns)

	# import pdb
	# pdb.set_trace()

	y = df['r'] # = target (specific growth rate, SGR)
	# remove gross growth rate, and string-valued columns:
	df = df.drop(columns=['r', 'Date', 'source', 'J_G'])

	# Arrange Data into Features Matrix and Target Vector
	X = df[df.columns[:-2]] # up to idx -3 = seaweed time series data

	# Split the data into training and testing sets
	X_train, X_test, Y_train, Y_test = train_test_split(X, y, random_state=0)

	# Random Forests in `scikit-learn` (with N = 100)
	rf = DecisionTreeRegressor(max_depth = 4,
								random_state = 0)
	rf.fit(X, y)

	fn = df.columns[:-2]
	cn = 'SGR'

	text_representation = tree.export_text(rf)
	print(text_representation)

	fig = plt.figure(figsize=(25,20))
	_ = tree.plot_tree(rf, feature_names=fn, filled=True)

	fig.savefig(f'rf_individualtree_{label}.png')
	dot = tree.export_graphviz(rf, out_file=None, feature_names=fn, filled=True)
	src = graphviz.Source(dot, format="png")
	src.render(f'graphviz_{label}.png')

	from dtreeviz.trees import dtreeviz # remember to load the package

	viz = dtreeviz(rf, X, y, target_name="SGR", feature_names=fn)
	viz.save(f'dtreeviz_{label}.svg')


data_path = os.path.join('.', 'df_all_no_gaussian.csv')
label = 'all'
graphRF(data_path, label=label)