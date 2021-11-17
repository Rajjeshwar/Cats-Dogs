# Image classification using CNN and K-fold cross validation


First, here's an excellent explanation from Colah's blog post: 

![image](https://user-images.githubusercontent.com/80246631/142145868-6f80d499-2fa2-440c-9b15-8c92d7f6b6e9.png)

Essentially, here we implement a 2D Convolution network to classify images of dogs and cats.

## Design decisions:

1. For optimization, we used a standard ADAM optimizer with default values for learning rate(0.001), beta1(0.9) and beta2(0.99).
2. Weights have been initialized using the default Xavier GLOROT scheme.
3. The loss function used is the binary cross entropy function alongside a sigmoid activation to get values in the range of {0, 1} representing the two classes.
4. To cross validate, a stratified K-fold validation scheme has been used instead of a standard train-val split. This helped in detecting and reducing overfitting in the model.
5. To further reduce overfitting, dropout layers were used. Using L2 norm did not help in alleviating the problem of overfitting by a noteworthy margin. Using dropout however, boosted validation accuracy by almost 5%.

## Evaluation: 

We use a binary cross entropy loss function.

![image](https://user-images.githubusercontent.com/80246631/142148427-d63b2c4b-427b-48f4-8f6f-db43a0650fe7.png)

The cross entropy loss is a very good measure of how distinguishable two discrete probability distributions are from each other. The sum of the probabilities of predictions add upto 1.

Furthermore, cross entropy is a very textbook loss for problems/formulations such as these. We use it over regression based losses like the mean squared error loss as we want to perform Convex optimization. The MSE function is non-convex for binary classification. In simple terms, if a binary classification model is trained with MSE Cost function, it is not guaranteed to minimize the Cost function. This is because MSE function expects real-valued inputs in range(-∞, ∞), while binary classification models output probabilities in range(0,1) through the sigmoid/logistic function.

![image](https://user-images.githubusercontent.com/20723780/138416248-eddf6e62-eeef-4ccb-8b96-013c42ada084.png)

For the purpose of cross validation a 10 fold, stratified K-fold cross validation is performed. Hence, the training data is picked from a strata randomly with ten different hold-outs per fold.

Finally, we plot the cross validation accuracy across 10 folds and find the mean to get a validation accuracy score. From my observation going for this approach instead of a typical train-val split helped increase test accuracy by more than 10%.



## Metrics: 

 ```
 Train Accuracy: 95.72
 
 K-fold Validation accuracy: [87.82564997673035, 83.26653242111206, 81.71342611312866, 86.47294640541077, 85.9719455242157, 83.96793603897095, 82.35589265823364,  86.21553778648376, 86.56641840934753, 84.16039943695068]
 
 Mean Validation accuracy: 84.85166847705841
 
 Standard deviation: 1.9333284734781433
 
 Test Accuracy: 83.63
 ```
 ## Install Requirements: 
 
The following were used for making this program-

1. Tensorflow
2. sklearn
3. numpy
4. pandas
5. os module
6. unittest
 
 ```
 pip install -r requirements.txt
 ```
 
 The following link provides a good walkthrough to setup tensorflow:
 
  ```
https://towardsdatascience.com/tensorflow-gpu-installation-made-easy-use-conda-instead-of-pip-52e5249374bc

 ```
 
 
 ## Format code to PEP-8 standards (Important for contributing to the repo): 
 
 This repository is strictly based on *PEP-8* standards. To assert PEP-8 standards after editing your own code, use the following: 
 
 ```
 black CatsVSDogs-Dataload.py
 black Classifier_Model.py
 ```
 
If you wish to use change the dataset used here change the following to correctly reflect the directory in `CatsVSDogs-Dataload.py`:

`data_directory = r"C:\Users\Desktop\Desktop\JuPyter Notebooks\data\Cats&Dogs\PetImages"`

The directory names of the categories used for classification can be changed here:

`categories = ["Dog", "Cat"]`

NOTE: This was trained on a 2080Super using tensorflow GPU, images were converted to greyscale and then resized to fit vram constraints. Training will take longer on GPUs not running CUDA, on CPUs and if larger datasets are used.

### Reference: 

1. https://cs231n.github.io/convolutional-networks/
2. https://www.uksim.info/isms2016/CD/data/0665a174.pdf
3. https://towardsdatascience.com/what-is-stratified-cross-validation-in-machine-learning-8844f3e7ae8e
4. https://www.superdatascience.com/blogs/convolutional-neural-networks-cnn-softmax-crossentropy

