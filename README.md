## Book Rating Prediction

The objective of this program is to develop a recommender system for an medium-sized dataset.
For this program, we are asked to predict the rating that a user will give to a book given their past book ratings.

### My Implementation
After trial error with various K-means based algorithms, matrix factorization was found to provide better results. Surprise library has in-built matrix factorization function that can be used.

```
from surprise import SVD
```

### Data Description
For this program we will use recommender system algorithms on the provided training dataset (train.csv) to traing a model that will be used to predict the ratings of a user for a book.
- train.csv - the training set (User, Book, Rating)
- test.csv - the test set (User, Book)
