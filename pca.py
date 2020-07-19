import numpy as np
import pandas as pd
import pprint
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pylab as pl
from sklearn.preprocessing import StandardScaler


iris = load_iris()
iris_df = pd.DataFrame(iris.data,columns=[iris.feature_names])
z=iris_df.head()
X = iris.data

y=X.shape
X_std = StandardScaler().fit_transform(X)

# print(iris_df)
# print(X)
print(z)
print(y)
print(X_std[0:4])

tra= X_std[0:4].T   # used to find out transpose it has to be done after standardisation
X_covariance_matrix = np.cov(X_std.T)

# print(tra)
print("the covar of mat is\n" ,X_covariance_matrix)


eig_vals, eig_vecs = np.linalg.eig(X_covariance_matrix)   #linear algebra used to find out eigen values

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)


# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])

tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
print ("Variance captured by each component is \n",var_exp)
print(40 * '-')
print ("Cumulative variance captured as we travel each component \n",cum_var_exp)

print ("All Eigen Values along with Eigen Vectors0")
pprint.pprint(eig_pairs)
print(40 * '-')
matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1),
                      eig_pairs[1][1].reshape(4,1)))

print ('Matrix W:\n', matrix_w)

Y = X_std.dot(matrix_w)
print (Y[0:5])

pl.figure()
target_names = iris.target_names
y = iris.target
for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
    pl.scatter(Y[y==i,0], Y[y==i,1], c=c, label=target_name)
pl.xlabel('Principal Component 1')
pl.ylabel('Principal Component 2')
pl.legend()
pl.title('PCA of IRIS dataset')
pl.show()
