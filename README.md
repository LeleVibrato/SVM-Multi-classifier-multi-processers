# SVM-Multi-classifier-multi-processes
Implemented a multi-class SVM that supports multi-core parallel computing


In this project, we have developed a multi-class Support Vector Machine (SVM) capable of harnessing multicore parallel processing. Applied to the task of handwritten digit classification, this classifier not only demonstrates remarkable training and prediction velocity but also achieves superior accuracy. The multi-class SVM employs a One-vs-One (OvO) decomposition strategy, and facilitates the use of kernel tricks. In one of the assessments, with a dataset characterized by 256 features, 10 categories, 8,566 training instances, and 2,432 test instances, the classifier, when employing the Gaussian kernel, has been observed to secure an accuracy exceeding 90% on the test set. Furthermore, the adoption of parallel computing significantly elevates performance, offering a speed enhancement of over 100% compared to single-threaded computations.
