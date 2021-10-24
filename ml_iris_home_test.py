
# ==========================================================================
# File Name: ml_iris_home_test
# Discription: This file contain a solution for bootcamp iris db home test.
#             It use the scikit-learn lib for clustering the iris db.
# =========================================================================

#%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns  # for plot styling

from sklearn import datasets
from sklearn.decomposition import PCA  # 1. Choose the model class
from sklearn import mixture #from sklearn.mixture import GMM  # 1. Choose the model class


def ml_iris_db_home_test():

    # step 1: Load the Iris db and plot it
    # --------------------------------------
    if 0:
        iris = datasets.load_iris()
        X_iris = iris.data[:, :2]  # we only take the first two features.
        y_iris = iris.target
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
    else:
        iris = sns.load_dataset('iris')

        sns.set()
        sns.pairplot(iris, hue='species', size=1.5);
        X_iris = iris.drop('species', axis=1)
        y_iris = iris['species']

    # Step 2: Methood 1, principal component analysis (PCA) to cluster the iris db
    # ----------------------------------------------------------------------------
    model = PCA(n_components=2)  # 2. Instantiate the model with hyperparameters
    model.fit(X_iris)  # 3. Fit to data. Notice y is not specified!
    X_2D = model.transform(X_iris)

    iris['PCA1'] = X_2D[:, 0]
    iris['PCA2'] = X_2D[:, 1]
    sns.lmplot("PCA1", "PCA2", hue='species', data=iris, fit_reg=False);

    # Step 3: Method 2 use Gaussian mixture model(GMM) to cluster the iris db
    # -----------------------------------------------------------------------
    #model = GMM(n_components=3, covariance_type='full')  # 2. Instantiate the model with hyperparameters
    model = mixture.GaussianMixture(n_components=3, covariance_type='full')
    model.fit(X_iris)  # 3. Fit to data. Notice y is not specified!
    y_gmm = model.predict(X_iris)  # 4. Determine cluster labels

    iris['cluster'] = y_gmm
    sns.lmplot("PCA1", "PCA2", data=iris, hue='species', col='cluster', fit_reg=False)


if __name__ == '__main__':

    ml_iris_db_home_test()

    print('end')