# SVM-Multi-classifier-multi-processers
Implemented a multi-class SVM that supports multi-core parallel computing

In this project, we have developed a multi-class Support Vector Machine (SVM) capable of harnessing multicore parallel processing. Applied to the task of handwritten digit classification, this classifier not only demonstrates remarkable training and prediction velocity but also achieves superior accuracy. The multi-class SVM employs a One-vs-One (OvO) decomposition strategy, and facilitates the use of kernel tricks. In one of the assessments, with a dataset characterized by 256 features, 10 categories, 8,566 training instances, and 2,432 test instances, the classifier, when employing the Gaussian kernel, has been observed to secure an accuracy exceeding 90% on the test set. Furthermore, the adoption of parallel computing significantly elevates performance, offering a speed enhancement of over 100% compared to single-threaded computations.


Additional Information:
1. Implemented an interface for selecting the type of kernel function.
2. The demo uses the Iris flower dataset and can be executed directly with the command ```python nlsvm_ovo_poly_kernel_multi_process.py```.
3. ```zip.train``` and ```zip.test``` are the training samples and test samples of the handwritten digit dataset, respectively.
4. The quadratic programming solver used is ```osqp```
5. The testing environment is ```python 3.9.7```, and it requires a minimum of```python 3.9```
6. The open-source license used is ```MIT License```

example：
```python
if __name__ == '__main__':
    from datetime import datetime
    import os
    from sklearn.model_selection import train_test_split
    import pandas as pd

    start_time = datetime.now()

    # Change directory to the script file location
    os.chdir(os.path.dirname(__file__))

    # Load training and testing data
    data_train = pd.read_csv(
        "zip.train", delimiter=" ", header=None).to_numpy()
    data_test = pd.read_csv("zip.test", delimiter=" ", header=None).to_numpy()
    X_train = data_train[:, 1:].astype(float)
    y_train = data_train[:, 0].astype(int).astype(str)
    X_test = data_test[:, 1:].astype(float)
    y_test = data_test[:, 0].astype(int).astype(str)

    print("Training on the handwritten digits dataset...")
    model = nlsvm(X_train, y_train, C=10.0, degree=6)
    print("Testing the classifier...")
    y_predict = predict(model, X_test)
    print(f"Test set accuracy: {accuracy(y_predict, y_test) * 100:.2f}%\n")

    end_time = datetime.now()
    print("Total time taken: ", (end_time - start_time).seconds, "seconds\n")

```


```
Training on the handwritten digits dataset...
Categories:  ['0' '1' '2' '3' '4' '5' '6' '7' '8' '9']
A total of 45 classifiers are required for this task...
Starting processes...
Classifier training completed: 3/45 ['0', '3']
Classifier training completed: 4/45 ['0', '4']
Classifier training completed: 5/45 ['0', '5']
Classifier training completed: 2/45 ['0', '2']
Classifier training completed: 8/45 ['0', '8']
......
Classifier training completed: 42/45 ['6', '9']
Classifier training completed: 43/45 ['7', '8']
Classifier training completed: 44/45 ['7', '9']
Classifier training completed: 45/45 ['8', '9']
Training complete!
Number of training samples: 7291; Dimensions: 256; Number of classes: 10
Testing the classifier...
Number of test samples: 2432
Test set accuracy: 94.82%

Total time taken:  352 seconds
```
