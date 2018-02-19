import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

class ProfanityClassifier:
    """ Example usage:
    
    # Use classifier
    pcf = ProfanityClassifier(verbose=True)

    # Load data
    trainPd = pd.read_csv("train.csv")
    testPd = pd.read_csv("test.csv")

    # Show data stats
    # pcf.show_data_stats(trainPd, testPd)

    # Create binary training data
    binaryTrainingData = pcf.create_binary_training_data(trainPd)

    # Train models
    startTime = time.time()
    gscvScores, featuresPds, vectorizers = pcf.train_on_binary_training_data(binaryTrainingData)
    elapsedTime = time.time() - startTime
    print("Training finished in {0}s".format(elapsedTime))

    # Show most important feature importances (for each model)
    # pcf.show_feature_importances(gscvScores)

    # Extra validation on remaining test set
    startTime = time.time()
    roc_auc, validatePd, predictedValidation = pcf.validate_on_remaining_train_data(trainPd, 
                                                                                    binaryTrainingData, 
                                                                                    vectorizers, 
                                                                                    gscvScores)
    elapsedTime = time.time() - startTime
    print("Avg ROC AUC on our validation data is {0}. Calculated in {1}s".format(roc_auc, elapsedTime))

    # Allow for simple visual inspection on extra validation results
    # pcf.print_random_classifications(validatePd,predictedValidation,n=100)

    # Predict on test set
    predicted = pcf.predict(testPd, 
                            vectorizers, 
                            gscvScores)

    # Save prediction
    pcf.save_predictions(testPd,predicted,"out.csv")
    """
    def __init__(self, verbose=False):
        self.verbose = verbose
        
    def show_text_len_values(self, column, label):
        """ Given column, calculates numbers such as
        mean, median, std dev, min, max text length 
        of the character counts of the text
        and prints these. Also prints a histogram based 
        on the character counts of the texts
        """
        textLen = column.apply(len)
        print("{7}:\n\tmin:\t{0}\n\tmax:\t{1}\n\tmedian:\t{2}\n\tmean:\t{3}\n\tstddev:\t{4}\n\t10th perc:\t{5}\n\t90th perc:\t{6}".format(
            np.min(textLen),
            np.max(textLen),
            np.median(textLen),
            round(np.mean(textLen),2),
            round(np.std(textLen),2),
            round(np.percentile(textLen,10),2),
            round(np.percentile(textLen,90),2),
            label
        ))

        # Show hist
        plt.hist(textLen, bins=50)
        plt.yscale('log', nonposy='clip')
        plt.ylabel('#texts (log)');
        plt.xlabel('text length');
        plt.show()

    def show_data_stats(self, trainPd, testPd):
        """ Shows/prints information on the dataset
        """
        # Amount of rows?
        print("Train data:\n\t#rows: {0}\n\tcolumns: {1}\n".format(trainPd.shape[0], list(trainPd.columns)))
        print("Test data:\n\t#rows: {0}\n\tcolumns: {1}\n".format(testPd.shape[0], list(testPd.columns)))

        # Can one line have multiple true labels?
        multipleLables = trainPd[np.sum(trainPd[['toxic','severe_toxic','obscene','threat','insult','identity_hate']], axis=1) > 1]
        print("Can one line have multiple true labels? (train)")
        print("\t#rows with more than one true label: {0}\n".format(multipleLables.shape[0]))

        # How many of each label is there?
        print("How many of each label is there?")
        for label in ['toxic','severe_toxic','obscene','threat','insult','identity_hate']:
            sumL = np.sum(trainPd[label], axis=0)
            print("\t#{0}:\t{1}\t(={2}% of train data)".format(label, sumL, round((sumL/trainPd.shape[0])*100,2)))

        # Also print amount of normal samples
        sumL = trainPd[np.sum(trainPd[['toxic','severe_toxic','obscene','threat','insult','identity_hate']], axis=1) < 1].shape[0]
        print("\t#normal:\t{0}\t(={1}% of train data)\n".format(sumL, round((sumL/trainPd.shape[0])*100,4)))

        # What is the mean, median, std dev, min, max text length? (train and test)
        self.show_text_len_values(trainPd['comment_text'],"Train")
        self.show_text_len_values(testPd['comment_text'],"Test")

        # What is the mean, median, std dev, min, max message length per class? 
        #  (ignoring the fact that a message can have multiple classes)
        for label in ['toxic','severe_toxic','obscene','threat','insult','identity_hate']:
            column = trainPd[trainPd[label] == 1]['comment_text']
            self.show_text_len_values(column,"Train {0} == 1".format(label))

    def create_binary_training_data(self, 
                                    trainPd, 
                                    classes=['toxic','severe_toxic','obscene','threat','insult','identity_hate'], 
                                    n=1000):
        """ Creates binary training data
        Returns dictionary with, for each class in classes an dataframe df
        where df has two classes of n values. Given class and "other" (True = given, False = other)
        """
        trainPds = {}

        # Create train datasets for each class. Resulting in two types of rows *class and *other
        # Bytheway we're not going to classify anything as "normal". It results from "other" in all binary classifications
        for label in classes:
            # Sample "label"(if len(label) < n, sample w/o replacement. Otherwise with replacement)
            ofLabel = (trainPd[label] == 1)
            replace = True
            if sum(ofLabel) > n:
                replace = False
            classPd = trainPd[ofLabel].sample(n, replace=replace)
            classPd['label'] = True

            # Sample not "label" #TODO: Stratified sampling? Or an even amount per class?
            # This class will contain LOTS (90%) of "normal" values if we don't do anything about it
            nonClassPd = trainPd[trainPd[label] != 1].sample(n, replace=False)
            nonClassPd['label'] = False

            # Add as one (vertically joined) dataframe to trainPds
            trainPds[label] = pd.concat([classPd, nonClassPd], axis=0)

            # We now have, per label, one trainPd with samples of given label and samples of all other labels
            # Note, the True values belong to the label class. But can ALSO belong to other classes.
            #       the False values DON'T belong to the label class. But to any or no amount of other classes 
            #       (With same distribution as the original dataset)
        return trainPds

    def calculate_simple_features(self, message):
        """Cleans message, calculates values on message such as:
        - % of capitals
        - #characters
        - #words
        - #punctuation
        - #!
        - #?
        etc...

        Returns pandas series of dictionary with features
        """
        return pd.Series({
            'numCapitals':sum(message.count(x) for x in ('Q','W','E','R','T','Y','U','I','O','P','A','S','D','F','G','H','J','K','L','Z','X','C','V','B','N','M')),   # RF model will scale/normalize in respect to num characters
            'numCharacters':len(message),
            'numWords':len(message.replace('\n', ' ').replace('  ', ' ').split(' ')),  # Not entirely correct, close enough
            'numPunctuation':sum(message.count(x) for x in ('!','@','#','$','%','^','&','*','(',')','.',',','/','\\',']','[','{','}','"',':',';',"'",']','`','~','|')),
            'numExclamation':message.count('!'),
            'numQuestion':message.count('?'),
        })

    def create_document_term_matrix(self, 
                                    messages, 
                                    max_features=1000, 
                                    strip_accents='unicode',  #None
                                    analyzer='word',
                                    ngram_range=(1,5),        #Optimize
                                    stop_words='english',
                                    lowercase=True,
                                    max_df=0.9,               #Optimize
                                    min_df=1,                 #Optimize
                                    tfidf=False):
        """ Create a document term matrix from given list of messages
        """
        vectorizer = CountVectorizer(max_features=max_features,
                                    strip_accents=strip_accents,
                                    analyzer=analyzer,
                                    ngram_range=ngram_range,
                                    stop_words=stop_words,
                                    lowercase=lowercase,
                                    max_df=max_df,
                                    min_df=min_df,
                                    )

        if tfidf:
            vectorizer = TfidfVectorizer(max_features=max_features,
                                    strip_accents=strip_accents,
                                    analyzer=analyzer,
                                    ngram_range=ngram_range,
                                    stop_words=stop_words,
                                    lowercase=lowercase,
                                    max_df=max_df,
                                    min_df=min_df)    # Override. Use the tfidf vectorizer
        vectorizer.fit(messages)

        dtm = vectorizer.transform(messages)
        dtmDf = pd.DataFrame(dtm.toarray(), columns=vectorizer.get_feature_names())

        return dtmDf, vectorizer

    def calculate_features(self, 
                           textPd, 
                           max_dtm_features=1000, 
                           vectorizers=None):
        """ Calculates features based on text column of dataframe
        Returns features and a dictionary of built vectorizers
        """
        # Simple
        if self.verbose: print("Creating simple features..")
        simpleDf = textPd['comment_text'].apply(self.calculate_simple_features)
        simpleDf = simpleDf.reset_index()
        simpleDf = simpleDf.drop(columns=['index'])

        # Based on vectorizers given or not given, calculate or use
        if vectorizers == None:
            if self.verbose: print("No vectorizers given. Creating new on data.")

            # DTM
            if self.verbose: print("Creating document term matrix..")
            if self.verbose: print("Count..")
            dtmCountDf, dtmCountVectorizer = self.create_document_term_matrix(list(textPd['comment_text']), max_features=max_dtm_features)

            # TF/IDF
            if self.verbose: print("TF/IDF..")
            dtmTfIdfDf, tfIdfVectorizer = self.create_document_term_matrix(list(textPd['comment_text']), max_features=max_dtm_features, tfidf=True)
        else:
            if self.verbose: print("Vectorizers found. Only applying new data on these vectorizers.")

            # DTM
            if self.verbose: print("Count..")
            dtmCount = vectorizers['dtmCountVectorizer'].transform(list(textPd['comment_text']))
            dtmCountDf, dtmCountVectorizer = pd.DataFrame(dtmCount.toarray(), columns=vectorizers['dtmCountVectorizer'].get_feature_names()), vectorizers['dtmCountVectorizer']

            # TF/IDF
            if self.verbose: print("TF/IDF..")
            dtmTfIdf = vectorizers['tfIdfVectorizer'].transform(list(textPd['comment_text']))
            dtmTfIdfDf, tfIdfVectorizer = pd.DataFrame(dtmCount.toarray(), columns=vectorizers['tfIdfVectorizer'].get_feature_names()), vectorizers['tfIdfVectorizer']

        # Merge into one feature DF
        featureDf = pd.concat([simpleDf, dtmCountDf, dtmTfIdfDf], axis=1)
        return featureDf, {'dtmCountVectorizer':dtmCountVectorizer, 'tfIdfVectorizer':tfIdfVectorizer}

    def train_on_binary_training_data(self, 
                                      binaryTrainingData,
                                      labels=['toxic','severe_toxic','obscene','threat','insult','identity_hate'],
                                      max_dtm_features=100):
        """ Train an classifier for each of the binary classes in binaryTrainingData
        """
        gscvScores = {}       # To hold results
        featuresPds = {}      # To hold dataframes with features
        vectorizers = {}      # To hold vectorizers necessary for testing
        for label in labels: 
            if self.verbose: print("Processing label {0}:\n------------------".format(label))
            currentPd = binaryTrainingData[label]
            textPd = currentPd[['id','comment_text']]
            originalLabelsPd = currentPd[labels]
            labelsPd = currentPd['label']

            # Calculate features
            featureDf, vectorizers[label] = self.calculate_features(textPd, max_dtm_features)
            featuresPds[label] = featureDf
            if self.verbose: print("\n#Features: {0}".format(featureDf.shape[1]))

            # Parameter optimization
            parameters = {
                'n_estimators': [8, 10, 12], 
                'max_depth': [None, 5, 10],
                'max_features': ['auto','log2',0.25,50] #,25
            }

            # Use random forest
            # grid search on given parameters
            # Use 10 fold cross validation
            if self.verbose: print("Grid search, cross validation..")
            rf = RandomForestClassifier()
            gridSearchCV = GridSearchCV(rf, parameters, cv=10)
            cvs = gridSearchCV.fit(featureDf, labelsPd)

            # Best parameters and corresponding CV score
            # When necessary, more results in: cvs.cv_results_
            if self.verbose: print("\n--- Results for {0} ---".format(label))
            if self.verbose: print("Best parameters: {0}".format(cvs.best_params_))
            if self.verbose: print("With best score: {0}".format(cvs.best_score_))
            if self.verbose: print("-------------------------\n".format(label))

            # Append scores to history
            gscvScores[label] = {
                'score':cvs.best_score_,
                'params':cvs.best_params_,
                'max_features':max_dtm_features,
                'best_estimator':cvs.best_estimator_
            }

        return gscvScores, featuresPds, vectorizers

    def show_feature_importances(self, gscvScores, top_n=30):
        """ Look into feature importances. Does it make sense?
        Allows for visual inspection
        """
        featureImportances = {}
        for label in ['toxic','severe_toxic','obscene','threat','insult','identity_hate']: 
            fi = gscvScores[label]['best_estimator'].feature_importances_
            featureImportances[label] = pd.DataFrame([fi], columns=featuresPds[label].columns).T.sort_values(by=0, ascending=False)
            if self.verbose: print("Feature importances for {0}:\n\n{1}\n\n".format(label, featureImportances[label][0:top_n]))

    def predict(self, 
                inPd, 
                vectorizers,
                gscvScores,
                labels=['toxic','severe_toxic','obscene','threat','insult','identity_hate']):
        """ Classify on given DF using 
        the "best estimators" for each class
        and the precalculated (on train data, without labels) vectorizers

        DF should contain column 'comment_text'

        Returns prediction (continuous, 0-1) for each of the labels to be True
        """
        predicted = {}
        for label in labels: 
            if self.verbose: print("Predicting on label {0}".format(label))

            # Calculate features on data
            # There is double work being done here 
            #   (for example for each label the simple features are extracted)
            #   This should be moved to a different location (or memoized)
            if self.verbose: print("Calculating features..")
            features, vectorizer = self.calculate_features(inPd, max_dtm_features=500, vectorizers=vectorizers[label])

            # Predict on previously trained models
            if self.verbose: print("Predicting..")
            estimator = gscvScores[label]['best_estimator']
            predicted[label] = estimator.predict_proba(features)
        if self.verbose: print("Done")
        return predicted

    def validate_on_remaining_train_data(self, 
                                         trainPd, 
                                         binaryTrainingData,
                                         vectorizers,
                                         gscvScores,
                                         sample=10000,
                                         labels=['toxic','severe_toxic','obscene','threat','insult','identity_hate']):
        """ Uses remaining training data 
        predicts using trained models
        calculates and shows average area under curve roc result

        Returns average AUC ROC and for each class the AUC ROC
        """
        # Create a dataframe from the train dataframe 
        # without all samples on which the models are trained
        trainedWith = pd.Series()
        for label in labels:
            trainedWith = pd.concat([trainedWith, binaryTrainingData[label]['id']], axis=0)    
        validatePd = trainPd[~trainPd['id'].isin(trainedWith)]

        # Validate on all train samples which have not been used in training
        if sample is not None:
            samplePd = validatePd.sample(sample)
        else:
            # Use the entire validation Pd
            samplePd = validatePd
        predictedValidation = self.predict(samplePd,vectorizers,gscvScores)

        # Calculate average area under roc
        resultPd = pd.Series()
        for label in labels:
            labelResult = pd.DataFrame(predictedValidation[label], columns=['not_{0}'.format(label),label])
            resultPd = pd.concat([resultPd, labelResult[label]], axis=1)
        roc_auc = roc_auc_score(samplePd[labels], resultPd[labels])

        return roc_auc,samplePd,predictedValidation

    def save_predictions(self, 
                         inPd, 
                         predictions,
                         csvPath,
                         labels=['toxic','severe_toxic','obscene','threat','insult','identity_hate']):
        """ Save results to output csv
        """
        resultPd = inPd['id']
        for label in labels:
            labelResult = pd.DataFrame(predicted[label], columns=['not_{0}'.format(label),label])
            resultPd = pd.concat([resultPd, labelResult[label]], axis=1)
        resultPd.to_csv(csvPath, index=False)

    def print_random_classifications(self, 
                                     actual,
                                     predicted,
                                     labels=['toxic','severe_toxic','obscene','threat','insult','identity_hate'],
                                     n=10):
        """Shows n messages with their classification
        """
        resultPd = actual
        resultPd = resultPd.reset_index()
        resultPd.drop(0)
        for label in labels:
            labelResult = pd.DataFrame(predicted[label], columns=['predicted_not_{0}'.format(label),'predicted_{0}'.format(label)])
            resultPd = pd.concat([resultPd, labelResult], axis=1)
        resultPd = resultPd.sample(n)

        for index, row in resultPd.iterrows():
            print("===================\n")
            print("Class:\t\tPredicted\tActual")
            print("Toxic:\t\t{0}\t\t{1}".format(round(row['predicted_toxic'],2),row['toxic']))
            print("severe_toxic:\t{0}\t\t{1}".format(round(row['predicted_severe_toxic'],2),row['severe_toxic']))
            print("obscene:\t{0}\t\t{1}".format(round(row['predicted_obscene'],2),row['obscene']))
            print("threat:\t\t{0}\t\t{1}".format(round(row['predicted_threat'],2),row['threat']))
            print("insult:\t\t{0}\t\t{1}".format(round(row['predicted_insult'],2),row['insult']))
            print("identity_hate:\t{0}\t\t{1}\n".format(round(row['identity_hate'],2),row['identity_hate']))
            print(row['comment_text'])
            print("\n===================\n")