#  Multiclass Multilabel prediction For stack overflow Questions

Goal: Given text for Questions , predict tags associated with them 

Important Note:  Model was trained with top 10 most occurring tags. You can exapnd the tags and retrain model.

 
 Please use below link to download preprocessed data file and save it in project folder.
 https://drive.google.com/file/d/1CusX1nxFgASPLqly2WMgfYy-AokzBeh9/view?usp=sharing
 
 Please use below link to download pretrained model and save it in project folder.
 https://drive.google.com/file/d/1ecDUPLZ8R4hYM-yhRrBif-LTi_2QPMN1/view?usp=sharing
 
 Model comparision:
 
 Bidirectional LSTM with pre trained Glove embeddings:
 
 Confusion matrix for label android:  
[[5527 3224]  
 [ 671  578]]  
Confusion matrix for label c#:  
[[7957  775]  
 [1166  102]]  
Confusion matrix for label c++:  
[[8986  341]  
 [ 639   34]]  
Confusion matrix for label html:  
[[8701  207]  
 [1078   14]]  
Confusion matrix for label ios:  
[[8880  364]  
 [ 737   19]]  
Confusion matrix for label java:  
[[7901  614]  
 [1371  114]]  
Confusion matrix for label javascript:  
[[7185 1098]  
 [1384  333]]  
Confusion matrix for label jquery:  
[[8803  216]  
 [ 967   14]]  
Confusion matrix for label php:  
[[8221  500]  
 [1094  185]]  
Confusion matrix for label python:  
[[8443  494]  
 [1019   44]]  
              precision    recall  f1-score   support

           0       0.15      0.46      0.23      1249
           1       0.12      0.08      0.10      1268
           2       0.09      0.05      0.06       673
           3       0.06      0.01      0.02      1092
           4       0.05      0.03      0.03       756
           5       0.16      0.08      0.10      1485
           6       0.23      0.19      0.21      1717
           7       0.06      0.01      0.02       981
           8       0.27      0.14      0.19      1279
           9       0.08      0.04      0.05      1063

   micro avg       0.16      0.12      0.14     11563
   macro avg       0.13      0.11      0.10     11563
weighted avg       0.14      0.12      0.12     11563
 samples avg       0.13      0.13      0.13     11563

roc_auc_score : 0.5102566074016955

XL Net model:

Confusion matrix for label android:
[[8585  106]
 [  32 1277]]
Confusion matrix for label c#:
[[8453   61]
 [ 156 1330]]
Confusion matrix for label c++:
[[9300   47]
 [  45  608]]
Confusion matrix for label html:
[[8985  155]
 [ 363  497]]
Confusion matrix for label ios:
[[9303   60]
 [  20  617]]
Confusion matrix for label java:
[[8263  140]
 [ 248 1349]]
Confusion matrix for label javascript:
[[8086  145]
 [ 493 1276]]
Confusion matrix for label jquery:
[[8850   73]
 [ 274  803]]
Confusion matrix for label php:
[[8421  166]
 [  70 1343]]
Confusion matrix for label python:
[[9108    9]
 [  57  826]]
              precision    recall  f1-score   support

           0       0.92      0.98      0.95      1309
           1       0.96      0.90      0.92      1486
           2       0.93      0.93      0.93       653
           3       0.76      0.58      0.66       860
           4       0.91      0.97      0.94       637
           5       0.91      0.84      0.87      1597
           6       0.90      0.72      0.80      1769
           7       0.92      0.75      0.82      1077
           8       0.89      0.95      0.92      1413
           9       0.99      0.94      0.96       883

   micro avg       0.91      0.85      0.88     11684
   macro avg       0.91      0.85      0.88     11684
weighted avg       0.91      0.85      0.88     11684
 samples avg       0.92      0.89      0.89     11684

roc_auc_score : 0.921757767916760

We are achieving better roc_auc score with XLnet model and hence we have selected it solve our problem statement.
 
 
