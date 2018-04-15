
import numpy as np
import pandas as pd
from sklearn import tree

df = pd.read_csv("PastHiresFinal.csv" , header = 0)
features = list(df.columns[:6])

y = df["Hired"]
X = df[features]
classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(X,y)

from IPython.display import Image  
from sklearn.externals.six import StringIO  
import pydot 

dot_data = StringIO()  
tree.export_graphviz(classifier, out_file=dot_data, feature_names=features)  
(graph, ) = pydot.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())

from sklearn.ensemble import RandomForestClassifier as RFC
classifier = RFC(n_estimators=10)
classifier = classifier.fit(X,y)

print(classifier.predict([10,1,4,0,0,0]))
print(classifier.predict([10,0,4,0,0,0]))