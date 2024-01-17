| Model         | VGG1     | VGG3    | VGG3 with aug | VGG16 transfer | MLP      |
| ------------- | --------| --------| --------------| ---------------| -------- |
| Training Time | 79.31   | 36.94   | 39.69         | 114.68         | 736.61   |
| Trainig loss  | 0.61    | 0.585   | 0.53          | 0.235          | 0.583    |
| Training accuracy | 68.125 | 78.75   | 80.257        | 91.875         | 76.25    |
| Testing accuracy  | 65     | 80      | 85            | 92.5           | 68.5     |
| Model parameters  | 40961153 | 10333505 | 10333505   | 17074241     | 62491137 |


Q.1. The results are more or less as expected. We can see that as the model becomes more and more complex our training loss increases and training accuracy also increases. Testing accuracy also follows the same trend as it increases as the model compelxity increases. We can see from the table that the training time also increases as the complexity in model is increased. One thing which doesn't fit is that the value of test accuracy is getting higher as compared to train accuracy. This might be the case because we are using very low amount of data for training and the testing data is quite similar to the data where our model is quite good fit. 

Q.2 Data augmentation can be helpful in improving the performance of deep learning models, including those based on the VGG architecture. Data augmentation techniques, such as rotating(used here), flipping, and cropping images, can increase the amount of training data available to the model, which can improve its ability to generalize to new, unseen data.
However, the effectiveness of data augmentation in improving the performance of a VGG model with only three layers may be limited. The original VGG architecture has 16 layers, and its success is largely due to its depth and the use of small convolutional filters, which allows it to learn more complex and abstract features. A VGG model with only three layers may not have enough capacity to effectively learn from augmented data, as it may not be able to capture the necessary complexity of the underlying data.
Therefore, while data augmentation may still be beneficial for a VGG3 model, its impact may not be as significant as it would be for a larger, more complex model like the original VGG.

Q.3 Yes, it does matter on how many epochs a model has been fine tuned. If the number of epochs are very high then you might have overfit on the training data thereby your modle is not generalizable and if you have very low number of epochs than your model is underfitting and there is a room for your model to be improved significantly.

Q.4
Yes, there are a few instances where our model is very confused as we have an image of dolphin in which it is half present inside water and half outside. Our model is not able to predict it as dolphin and wrongly classify it as a jellyfish as it seems to be a jellyfish.

Q.5
The MLP model with comparable parameters to VGG16 would likely have a low capacity for learning complex features and patterns in the data as it doesn't take into account the spatial behaviour that CNNs take into account.

Due to the large number of layers and neurons, this model is prone to overfitting, especially if the dataset is small or the regularization techniques are not properly implemented.

Compared to other models with fewer parameters, the MLP model requires more computational resources and time to train.