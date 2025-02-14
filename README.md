# Predict Visibility 

This is a simple coding challenge for a simple regression using method(s) from scikit.

The training data set (`train.csv`) is provided by the challenger, the last column is visibility. There are three caveats:

1. There is a column of dates, which will be read as a string and needs to be converted to the correct scalar date type for the regression to be possible.

2. There are null values in the input training data (all columns apart from the last one). There are regression methods that can handle this, you need to pick one of them for the library.

3. There are also null values in the input predicates - i.e. the visibility values in the training data. You need to identify and remove the corresponding rows from the training set.

Finally, you need to generate an output file (`submission.csv`) in the format specified by the challenger, i.e. like sample_submission.csv

There are two test data files for predictions, a small one (`test_tiny.csv`) and the full one (`test.csv`). The output of running the latter is the one to be submitted to the challenger for assessment. 
