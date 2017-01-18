# cs205_feature_selection
Project for CS205
#Problem Statement
Nearest neighbor algorithm is very simple, yet very competitive classification algorithm. But it
is very sensitive to irrelevant features which can degrade the accuracy of the classifier.
Given a Dataset we have to find out the most importance features present in the data. We have
to use following searches/methods to find out relevant features.
• Forward Selection
• Backward Elimination
• Our Original Algorithm.

#Feature Selection
Feature selection is the process of selecting a subset of relevant features for the use of model
construction.[1]
Feature selection techniques are used for following reasons:
• simplification of models to make them more interpretable.
• reduce training time.
• reduce overfitting.

In a large data set many features are either redundant or irrelevant and thus can be removed
without incurring much loss of information.
Two of the most important methods to select subset of relevant features is:
• Sequential forward selection.
• Sequential backward selection.
Sequential Forward Selection

In this method we start with empty set. Sequentially we add the features which maximizes the
accuracy when combined with the features that have already been selected. We do this till all
features are selected then we select the subset which resulted in best accuracy.
Implementation
I used fitcknn method for knn training, crossval for ”leave one out” validation and kfoldloss
to calculate the loss which is subtracted by 1.00 to give the accuracy of the classifier.

#Sequential Backward Selection
This method is similar but performed exactly opposite. we start with all features selected.
Sequentially we remove each features and check the accuracy and select that which maximizes
the accuracy when it is removed from the set. We do this till features set is empty and then we
select the subset which gave the best accuracy.

#Implementation
Same as in forward selection i used fitcknn method for knn training, crossval for ”leave one
out” validation and kfoldloss to calculate the loss which is subtracted by 1.00 to give the ac-
curacy of the classifier. I used horzcat method as well for the concatenation of two matrices
in horizontal manner.

#My Solutions
Even though nearest neighbor is quite powerful but it is very slow because of number of com-
parisons done each points for classification. Computational cost grows exponentially with the
increase in dimensions. To overcome this issue I utilized the power of decision trees, I choose
boosting for this project.

I created decision tree for the whole data and then choose top 5 features of decision tree which it
is using to make decisions. Then I calculated accuracy of each of the features and then selected
top 2 features with better accuracy, since our data is strongly correlated to two features.
Implementation

I used fitensemble method to train a decision tree using X & Y. Then I used predictor Importance method to find out all the important decisions used in decision tree which gives a
floating number for all the features. After sorting that array in descending order I picked top
5 decisions and then using fitcknn I calculated the accuracy of each features and then sorted
the accuracy in descending order and picked top 2 features and calculated accuracy of that subset.

#Explanation
I tried to implement a method which reduces the time to find feature subset substantially thats
why I picked boosting(AdaBoost to be specific). Then I used boosting to find the important
features. The reason for improvement in time can be seen with respect to the forward selection
and backward elimination is due to reduction in training decision trees. Decision trees complex-
ity does not increases with the increase in features in data. AdaBoost training process selects
only those features known to improve the the model, reducing dimensionality and potentially
improving execution time as irrelevant features do not need to be computed.

#Testing
For this project I evaluated my implementation of all 3 algorithms on 6 different datasets:
cs 205 small52, cs 205 small64, cs 205 small65, cs 205 large52, cs 205 large64,
cs 205 large65. I evaluated feature selected after running the algorithms, accuracy of the final
feature selected and time taken to complete the specific algorithm.

#Time Analysis
For all the small data sets with 10 features Forward selection takes an average of approx 45
seconds to run, Backward elimination takes about 45 seconds to select features. My algorithm
for all 3 small datasets takes about 6 seconds to do feature selection.

For the large data sets with 100 features time increases exponentially. Forward selection takes
on an average about 1 hour to complete feature selection. Backward selection also takes about
1 hours and 15 minutes to complete on average. So backward elimination takes a little more
time for feature selection than forward selection. My algorithm performs the same as it for the
small datasets, it takes on average 6 seconds to do feature selection.

#Feature & Accuracy Analysis
For the small dataset forward feature selection resulted in subset of 2-3 features for all the data
sets. For backward elimination dataset 52 & 64 resulted in subset of 2 features whereas dataset
65 gave subset of 9 features. My algorithm only gives subset of 2 best features since we have
assumed that 2 feature is strongly co-related with the results.

For large dataset forward selection resulted in subset of different lengths dataset 52 gave subset
of 6 features, dataset 64 gave subset of 5 features and dataset 65 gave subset of 3 features. For
backward elimination all large data set resulted in vary large subset of features.

My algorithm gave 2 features for all large datasets. But the accuracy for all them was in par
with the accuracy from forward selection subset and backward elimination subset for dataset
65 & 64 except for dataset 52, where my algorithm took 6 seconds to select feature which gave
accuracy of 96% which was better than accuracy of the subsets selected from both forward
selection and backward elimination.

#Final Evaluation
Nearest neighbor algorithm is very powerful tool for the feature selection as well as for
classification. But with increase in the dimension computational cost increases exponentially.
So in conclusion we should use nearest neighbor algorithm when the number of features is less
for the dataset. If the number of features is quite large then we should explore other options
for feature selection for example decision trees, PCA etc.