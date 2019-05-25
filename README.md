## Book Rating Prediction

The objective of this program is to develop a recommender system for an medium-sized dataset.
For this program, we are asked to predict the rating that a user will give to a book given their past book ratings.

### Data Description
For this program we will use recommender system algorithms on the provided training dataset (train.csv) to traing a model that will be used to predict the ratings of a user for a book.
- train.csv - the training set (User, Book, Rating)
- test.csv - the test set (User, Book)

### My Implementation
After trial error with various K-means based algorithms, matrix factorization was found to provide better results. Surprise library has in-built matrix factorization function that can be used.

```
from surprise import SVD
```
The SVD function can be utilised in the following way:(This is my implementation)
```
algo=SVD(n_epochs=50,lr_all=0.01,reg_all =0.04,n_factors =250)

kf = KFold(n_splits=5)

for trainset, testset in kf.split(data):
    # train and test algorithm.
    algo.fit(trainingSet)
    predictions = algo.test(testset)
    accuracy.rmse(predictions, verbose=True)
```
### Instructions
For installing surprise, numpy library is a pre-requisite. Install numpy using
```
pip install numpy
```
You need to install the surprise library which can be done by using the following command
```
pip install scikit-surprise
```
or with
```
conda install -c conda-forge scikit-surprise
```
or
```
pip install numpy cython
git clone https://github.com/NicolasHug/surprise.git
cd surprise
python setup.py install
```
For more information and usgae instructions of the surprise library, [click here](http://surpriselib.com/).

### References:
- https://surprise.readthedocs.io/en/stable/index.html
- https://surprise.readthedocs.io/en/stable/prediction_algorithms.html
- https://surprise.readthedocs.io/en/stable/co_clustering.html#surprise.prediction_algorithms.co_clustering.CoClustering
- https://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVDpp
- https://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVD
- https://medium.com/datadriveninvestor/how-to-built-a-recommender-system-rs-616c988d64b2
- https://github.com/rashmishrm/Collaborative-Filtering-Demo
- https://cambridgespark.com/practical-introduction-to-recommender-systems/
- https://www.ethanrosenthal.com/2016/01/09/explicit-matrix-factorization-sgd-als/
- https://medium.com/@james_aka_yale/the-4-recommendation-engines-that-can-predict-your-movie-tastes-bbec857b8223
- https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-recommendation-engine-python/
- https://surprise.readthedocs.io/en/stable/matrix_factorization.html
- https://kerpanic.wordpress.com/2018/03/26/a-gentle-guide-to-recommender-systems-with-surprise/
- https://github.com/lppier/Recommender_Systems
- https://github.com/nikunjlad/Movie-Recommendation-System-Using-Surprise/blob/master/Movie%20Recommender%20System.ipynb
