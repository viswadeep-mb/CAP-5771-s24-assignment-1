import numpy as np
import utils as u
import new_utils as nu
from numpy.typing import NDArray
from typing import Any
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix,make_scorer, f1_score,accuracy_score, recall_score,precision_score

from sklearn.metrics import top_k_accuracy_score

from sklearn.utils.class_weight import compute_class_weight

from sklearn.model_selection import StratifiedKFold,cross_val_score
"""
   In the first two set of tasks, we will narrowly focus on accuracy - 
   what fraction of our predictions were correct. However, there are several 
   popular evaluation metrics. You will learn how (and when) to use these evaluation metrics.
"""


# ======================================================================
class Section3:
    def __init__(
        self,
        normalize: bool = True,
        frac_train=0.2,
        seed=42,
    ):
        self.seed = seed
        self.normalize = normalize

    def analyze_class_distribution(self, y: NDArray[np.int32]) -> dict[str, Any]:
        """
        Analyzes and prints the class distribution in the dataset.

        Parameters:
        - y (array-like): Labels dataset.

        Returns:
        - dict: A dictionary containing the count of elements in each class and the total number of classes.
        """
        # Your code here to analyze class distribution
        # Hint: Consider using collections.Counter or numpy.unique for counting

        uniq, counts = np.unique(y, return_counts=True)
        print(f"{uniq=}")
        print(f"{counts=}")
        print(f"{np.sum(counts)=}")

        for i, j in enumerate(counts):
            c[uniq[i]] = j
            
        return {
            "class_counts": c,  # Replace with actual class counts
            "num_classes": len(uniq)  # Replace with the actual number of classes
        }

    # --------------------------------------------------------------------------
    """
    A. Using the same classifier and hyperparameters as the one used at the end of part 2.B. 
       Get the accuracies of the training/test set scores using the top_k_accuracy score for k=1,2,3,4,5. 
       Make a plot of k vs. score for both training and testing data and comment on the rate of accuracy change. 
       Do you think this metric is useful for this dataset?
    """

    def partA(
        self,
        Xtrain: NDArray[np.floating],
        ytrain: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> tuple[
        dict[Any, Any],
        NDArray[np.floating],
        NDArray[np.int32],
        NDArray[np.floating],
        NDArray[np.int32],
    ]:
        """ """
        # Enter code and return the `answer`` dictionary

        answer = {}

        clf = LogisticRegression(max_iter=300, random_state=self.seed)
        clf.fit(Xtrain, ytrain)

        # Predict probabilities for training and test sets
        ytrain_pred = clf.predict_proba(Xtrain)
        ytest_pred = clf.predict_proba(Xtest)

        # Calculate top-k accuracy scores
        topk = [k for k in range(1, 6)]
        tuple_plot_scores_test = []
        tuple_plot_scores_train = []

        for k in topk:
            topk_dict = {}

            # Calculate top-k accuracy score for both training and test sets
            scores_train = top_k_accuracy_score(ytrain, ytrain_pred, normalize=True, k=k)
            scores_test = top_k_accuracy_score(ytest, ytest_pred, normalize=True, k=k)
            topk_dict["score_train"] = scores_train
            topk_dict["score_test"] = scores_test
            answer[k] = topk_dict
            tuple_plot_scores_test.append((k, scores_test))
            tuple_plot_scores_train.append((k, scores_train))

        # Store the trained classifier in the answer dictionary
        answer["clf"] = clf
        answer["plot_k_vs_score_train"] = tuple_plot_scores_train
        answer["plot_k_vs_score_test"] = tuple_plot_scores_test
        answer["text_rate_accuracy_change"] = "The model consistently demonstrates positive improvements in accuracy as the value of k increases for the testing data, suggesting that the model becomes increasingly proficient in predicting the top-k classes"
        answer["text_is_topk_useful_and_why"] = "The top-k accuracy metric is valuable for evaluating the model's performance as it assesses its capability to make accurate predictions across a broader spectrum of potential classes. This metric extends beyond conventional accuracy measures, providing a deeper understanding of the model's effectiveness in capturing relevant patterns."

        """
        # `answer` is a dictionary with the following keys:
        - integers for each topk (1,2,3,4,5)
        - "clf" : the classifier
        - "plot_k_vs_score_train" : the plot of k vs. score for the training data, 
                                    a list of tuples (k, score) for k=1,2,3,4,5
        - "plot_k_vs_score_test" : the plot of k vs. score for the testing data
                                    a list of tuples (k, score) for k=1,2,3,4,5

        # Comment on the rate of accuracy change for testing data
        - "text_rate_accuracy_change" : the rate of accuracy change for the testing data

        # Comment on the rate of accuracy change
        - "text_is_topk_useful_and_why" : provide a description as a string

        answer[k] (k=1,2,3,4,5) is a dictionary with the following keys: 
        - "score_train" : the topk accuracy score for the training set
        - "score_test" : the topk accuracy score for the testing set
        """

        return answer, Xtrain, ytrain, Xtest, ytest

    # --------------------------------------------------------------------------
    """
    B. Repeat part 1.B but return an imbalanced dataset consisting of 90% of all 9s removed.  Also convert the 7s to 0s and 9s to 1s.
    """

    def partB(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> tuple[
        dict[Any, Any],
        NDArray[np.floating],
        NDArray[np.int32],
        NDArray[np.floating],
        NDArray[np.int32],
    ]:
        """"""
        # Enter your code and fill the `answer` dictionary
        answer = {}

        #X, y, Xtest, ytest = u.prepare_data()
        X, y = u.filter_out_7_9s(X, y)
        Xtest, ytest = u.filter_out_7_9s(Xtest, ytest)
        
        X,y=nu.remove_90_9s(X,y)
        Xtest,ytest=nu.remove_90_9s(Xtest,ytest)
        
        X,y=nu.convert_7_0(X,y)
        Xtest,ytest=nu.convert_7_0(Xtest,ytest)
        
        X,y=nu.convert_9_1(X,y)
        Xtest,ytest=nu.convert_9_1(Xtest,ytest)

        answer["length_Xtrain"] = len(X)  
        answer["length_Xtest"] = len(Xtest)
        answer["length_ytrain"] = len(y)
        answer["length_ytest"] = len(ytest)
        answer["max_Xtrain"] = X.max()
        answer["max_Xtest"] = Xtest.max()

        # Answer is a dictionary with the same keys as part 1.B

        return answer, X, y, Xtest, ytest

    # --------------------------------------------------------------------------
    """
    C. Repeat part 1.C for this dataset but use a support vector machine (SVC in sklearn). 
        Make sure to use a stratified cross-validation strategy. In addition to regular accuracy 
        also print out the mean/std of the F1 score, precision, and recall. As usual, use 5 splits. 
        Is precision or recall higher? Explain. Finally, train the classifier on all the training data 
        and plot the confusion matrix.
        Hint: use the make_scorer function with the average='macro' argument for a multiclass dataset. 
    """

    def partC(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> dict[str, Any]:
        """"""

        # Enter your code and fill the `answer` dictionary
        answer = {}

        #partB_answer,X,y,Xtest,ytest = self.partB(X=X,y=y,Xtest=Xtest,ytest=ytest)

        clf=SVC(random_state=self.seed,kernel='linear')
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)

        scoring = {'f1': make_scorer(f1_score, average='macro'),
           'precision': make_scorer(precision_score, average='macro'),
           'recall': make_scorer(recall_score, average='macro'),
           'accuracy':'accuracy'}
        scores_cv = {metric: cross_val_score(clf, X, y, scoring=scoring[metric], cv=cv)
          for metric in scoring}
        
        scores_cv_stra={}
        scores_cv_stra['mean_accuracy']=np.mean(scores_cv['accuracy'])
        scores_cv_stra['mean_recall']=np.mean(scores_cv['recall'])
        scores_cv_stra['mean_precision']=np.mean(scores_cv['precision'])
        scores_cv_stra['mean_f1']=np.mean(scores_cv['f1'])
        scores_cv_stra['std_accuracy']=np.std(scores_cv['accuracy'])
        scores_cv_stra['std_recall']=np.std(scores_cv['recall'])
        scores_cv_stra['std_precision']=np.std(scores_cv['precision'])
        scores_cv_stra['std_f1']=np.std(scores_cv['f1'])
        
        answer["scores"]=scores_cv_stra
        answer['cv']=cv
        answer['clf']=clf

        clf.fit(X,y)
        y_pred_train=clf.predict(X)
        y_pred_test=clf.predict(Xtest)


        if scores_cv_stra['mean_precision'] > scores_cv_stra['mean_recall']:
            answer["is_precision_higher_than_recall"]= 'Precision'
        else:
            answer["is_precision_higher_than_recall"]= 'recall'

        #answer["explain_is_precision_higher_than_recall"] : 'a string'

        answer['confusion_matrix_train'] = confusion_matrix(y,y_pred_train)
        answer['confusion_matrix_test']  = confusion_matrix(ytest,y_pred_test)

        
        """
        Answer is a dictionary with the following keys: 
        - "scores" : a dictionary with the mean/std of the F1 score, precision, and recall
        - "cv" : the cross-validation strategy
        - "clf" : the classifier
        - "is_precision_higher_than_recall" : a boolean
        - "explain_is_precision_higher_than_recall" : a string
        - "confusion_matrix_train" : the confusion matrix for the training set
        - "confusion_matrix_test" : the confusion matrix for the testing set
        
        answer["scores"] is dictionary with the following keys, generated from the cross-validator:
        - "mean_accuracy" : the mean accuracy
        - "mean_recall" : the mean recall
        - "mean_precision" : the mean precision
        - "mean_f1" : the mean f1
        - "std_accuracy" : the std accuracy
        - "std_recall" : the std recall
        - "std_precision" : the std precision
        - "std_f1" : the std f1
        """

        return answer

    # --------------------------------------------------------------------------
    """
    D. Repeat the same steps as part 3.C but apply a weighted loss function (see the class_weights parameter).  Print out the class weights, and comment on the performance difference. Use the `compute_class_weight` argument of the estimator to compute the class weights. 
    """

    def partD(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> dict[str, Any]:
        """"""
        # Enter your code and fill the `answer` dictionary
        answer = {}

        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)

        clf= SVC(random_state=self.seed, kernel='linear', class_weight={0: class_weights[0], 1: class_weights[1]})

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)

        scoring = {
            'f1': make_scorer(f1_score, average='macro'),
            'precision': make_scorer(precision_score, average='macro'),
            'recall': make_scorer(recall_score, average='macro'),
            'accuracy': 'accuracy'
        }

        # Perform cross-validation
        scores_cv = {metric: cross_val_score(clf, X, y, scoring=scoring[metric], cv=cv)
                              for metric in scoring}

        # Calculate mean and standard deviation of scores
        scores_cv_wt = {}
        scores_cv_wt['mean_accuracy'] = np.mean(scores_cv['accuracy'])
        scores_cv_wt['mean_recall'] = np.mean(scores_cv['recall'])
        scores_cv_wt['mean_precision'] = np.mean(scores_cv['precision'])
        scores_cv_wt['mean_f1'] = np.mean(scores_cv['f1'])
        scores_cv_wt['std_accuracy'] = np.std(scores_cv['accuracy'])
        scores_cv_wt['std_recall'] = np.std(scores_cv['recall'])
        scores_cv_wt['std_precision'] = np.std(scores_cv['precision'])
        scores_cv_wt['std_f1'] = np.std(scores_cv['f1'])

        # Fit the classifier on the entire training data
        clf.fit(X, y)

        # Predict on training and testing data
        y_pred_train_wt= clf.predict(X)
        y_pred_test_wt = clf.predict(Xtest)

        # Generate confusion matrices
        confusion_matrix_train_wt = confusion_matrix(y, y_pred_train_wt)
        confusion_matrix_test_wt= confusion_matrix(ytest, y_pred_test_wt)

        answer["scores"] = scores_cv_wt
        answer['cv'] = cv
        answer['clf'] = clf
        answer['class_weights'] = class_weights
        answer['confusion_matrix_train'] = confusion_matrix_train_wt
        answer['confusion_matrix_test'] = confusion_matrix_test_wt
        answer['explain_purpose_of_class_weights'] = "The class weights are used to address class imbalance by penalizing misclassifications of the minority class more heavily."
        answer['explain_performance_difference'] = "The performance difference observed with class weights reflects the model's improved ability to generalize to the minority class, leading to more balanced performance metrics across all classes."

        """
        Answer is a dictionary with the following keys: 
        - "scores" : a dictionary with the mean/std of the F1 score, precision, and recall
        - "cv" : the cross-validation strategy
        - "clf" : the classifier
        - "class_weights" : the class weights
        - "confusion_matrix_train" : the confusion matrix for the training set
        - "confusion_matrix_test" : the confusion matrix for the testing set
        - "explain_purpose_of_class_weights" : explanatory string
        - "explain_performance_difference" : explanatory string

        answer["scores"] has the following keys: 
        - "mean_accuracy" : the mean accuracy
        - "mean_recall" : the mean recall
        - "mean_precision" : the mean precision
        - "mean_f1" : the mean f1
        - "std_accuracy" : the std accuracy
        - "std_recall" : the std recall
        - "std_precision" : the std precision
        - "std_f1" : the std f1

        Recall: The scores are based on the results of the cross-validation step
        """

        return answer
