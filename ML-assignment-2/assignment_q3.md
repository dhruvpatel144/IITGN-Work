------------------------------------------------------RIDO----------------------------------------------------------
Model Number: 1
|--- feature_0 <= 0.33
|   |--- class: 1
|--- feature_0 >  0.33
|   |--- class: 0

Model Number: 2
|--- feature_0 <= 0.16
|   |--- class: 0
|--- feature_0 >  0.16
|   |--- class: 1

Model Number: 3
|--- feature_0 <= 1.40
|   |--- class: 0
|--- feature_0 >  1.40
|   |--- class: 1

-------------------------------------------------Classification-----------------------------------------------------
Criteria : entropy
Accuracy:  0.7666666666666667
Precision:  0.7142857142857143
Recall:  0.5
Precision:  0.782608695652174
Recall:  0.9

Model Number: 1
|--- feature_1 <= 0.02
|   |--- class: 0
|--- feature_1 >  0.02
|   |--- class: 1

Model Number: 2
|--- feature_1 <= -0.87
|   |--- class: 0
|--- feature_1 >  -0.87
|   |--- class: 1


Model Number: 3
|--- feature_0 <= -0.91
|   |--- class: 1
|--- feature_0 >  -0.91
|   |--- class: 0


Accuracy from Adaboost: 0.75
Accuracy from stump: 0.6666666666666666
Precision for 1 in Adaboost :  1.0
Precision for 1 in stump :  1.0
Recall for  1 in Adaboost:  0.4
Recall for  1 in stump:  0.2
Precision for 0 in Adaboost :  0.7
Precision for 0 in stump :  0.6363636363636364
Recall for  0 in Adaboost:  1.0
Recall for  0 in stump:  1.0

The plots can be genrated by running the file q3_ADABoost.py.