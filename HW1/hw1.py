from  sklearn import neighbors, datasets
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap


#List of N for KNN
n_neighbors = [1,2,3,5,7]

data = {
    'Model Year': [2013, 2014, 2015, 2016, 2017, 2019, 2020, 2020],
    'Driving Range (miles)': [208, 208, 208, 210, 210, 270, 330, 337]
}
#List Color Map
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])


for n in n_neighbors:
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
    clf.fit(data['Driving Range (miles)'], data['Model Year'])
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    _, ax = plt.subplots()
    DecisionBoundaryDisplay.from_estimator(clf, data['Driving Range (miles)'], data['Model Year'], ax=ax)
    plt.show()

    