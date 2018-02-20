import math
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score


class ProfanityClassifier:
    """ Example usage:

    from ProfanityClassifier import ProfanityClassifier
    import pandas as pd

    pcf = ProfanityClassifier(verbose=True)

    # Train model on train data
    # Optional: numTrainSamplesPerClass (default: 1000), maxFeatures (default: 250)
    trainedModel = pcf.train("train.csv")

    # OR optionally load existing model
    #trainedModel = pcf.load_model("model.pickle")

    # Look into model metrics
    pcf.get_model_metrics(trainedModel)

    # Predict samples on model.
    # Possibilities:
    # - Samples are given in a csv (with rows of format: id, text\n)
    #   Returns dictionary or DataFrame depending on "dictOut"
    predictions = pcf.predict_on_csv(trainedModel, "test.csv", dictOut=True)

    # - Samples are given in a dictionary or list of dictionaries
    #   Returns dictionary or DataFrame depending on "dictOut"
    predictions = pcf.predict_on_dictionary(trainedModel, {
        'id':'1234',
        'comment_text':'Hi, this is a comment :)'
    }, dictOut=True)

    # - Samples are given in a pandas DataFrame:
    predictions = pcf.predict(trainedModel, pd.DataFrame([{
        'id':'1234',
        'comment_text':'Hi, this is a comment :)'
    }]))

    # And optionally save model
    #pcf.save_model(trainedModel, "model.pickle")
    """
    def __init__(self, verbose=False):
        self.verbose = verbose

    def train(self,
              csvFilePath,
              numTrainSamplesPerClass=500,
              maxFeatures=250):
        """ Trains model on csv in filePath
        Returns trained model
        """
        trainPd = pd.read_csv(csvFilePath)

        # Create binary training data
        binaryTrainingData = self.__create_binary_training_data(trainPd,
                                                                n=numTrainSamplesPerClass)

        # Train on binary training data
        maxDtmFeatures = int((maxFeatures-6)/2)
        gscvScores, vectorizers = self.__train_on_binary_training_data(binaryTrainingData,
                                                                       max_dtm_features=maxDtmFeatures)

        model_avg_roc_auc = self.__validate_on_remaining_train_data({'gscvScores': gscvScores,
                                                                     'vectorizers': vectorizers},
                                                                    trainPd,
                                                                    binaryTrainingData)

        # Return model
        # (basically a combination of the vectorizers and for every class one model (gscvScores))
        return {'gscvScores': gscvScores,
                'vectorizers': vectorizers,
                'model_avg_roc_auc': model_avg_roc_auc}

    def predict_on_csv(self,
                       model,
                       csvFilePath,
                       dictOut=True):
        """ Predict given csv with:
        id, comment_text, class1..,classn


        Returns dataframe (or dict when dictOut=True) with rows made up of:
           id, predictions (continuous, 0-1) for each of the classes
        """
        testPd = pd.read_csv(csvFilePath)
        predictions = self.predict(model, testPd)

        # Depending on required output, return DF or dict
        if dictOut:
            return predictions.T.to_dict()
        return predictions

    def predict_on_dictionary(self,
                              model,
                              testDict,
                              dictOut=True):
        """ Predict given dictionary with:
        id, comment_text, class1..,classn

        Returns dataframe (or dict when dictOut=True) with rows made up of:
           id, predictions (continuous, 0-1) for each of the classes
        """
        if type(testDict) is not list:
            testDict = [testDict]
        testPd = pd.DataFrame(testDict)
        predictions = self.predict(model, testPd)

        # Depending on required output, return DF or dict
        if dictOut:
            return predictions.T.to_dict()
        return predictions

    def predict(self, model, testPd):
        """ Classify on given dataframe using
        the "best estimators" for each class in the model
        and the precalculated vectorizers in the model

        DF should contain column 'comment_text'

        Returns dataframe with rows made up of:
           id, predictions (continuous, 0-1) for each of the classes
        """
        # Retrieve classes from model
        classes = list(model['gscvScores'].keys())

        # Predict rows in testPd on given model
        predicted = {}
        for cls in classes:
            if self.verbose:
                print("\n--- Predicting class {0} ---".format(cls))

            # Calculate features on data
            # There is double work being done here
            #   (for example for each label the simple features are extracted)
            #   This should be moved to a different location (or memoized)
            if self.verbose:
                print("Calculating features..")

            features, vectorizer = self.__calculate_features(testPd,
                                                             vectorizers=model['vectorizers'][cls])

            # Predict on previously trained models
            if self.verbose:
                print("Predicting..")

            estimator = model['gscvScores'][cls]['best_estimator']
            predicted[cls] = estimator.predict_proba(features)

            if self.verbose:
                print("-------------------------\n")
        if self.verbose:
            print("Done")

        # Process result to format: 'id', 'class1', 'class2', 'classn' etc..
        resultPd = testPd['id']
        resultPd = resultPd.reset_index()
        resultPd.drop(0)
        for cls in classes:
            labelResult = pd.DataFrame(predicted[cls], columns=['not_{0}'.format(cls), cls])
            resultPd = pd.concat([resultPd, labelResult[cls]], axis=1)

        return resultPd

    def save_model(self, model, destinationPath):
        """ Saves given model as a pickle to destinationPath
        Does a simple verification given model actually is
        from our classifier
        """
        # Check if model contains vectorizers, gscvScores
        if 'vectorizers' not in model or 'gscvScores' not in model:
                raise ValueError("""
                    Invalid model given.
                    Model should result from ProfanityClassifier().train(..)
                    """)
        else:
            # Model is ok. Pickle and write to file
            pickle.dump(model, open(destinationPath, 'wb'))

    def load_model(self, modelPath):
        """ Loads model from pickle at modelPath
        Does a simple verification to test it actually
        is a model resulting from our classifier
        """
        model = pickle.load(open(modelPath, 'rb'))

        # Check if model contains vectorizers, gscvScores
        if 'vectorizers' not in model or 'gscvScores' not in model:
                raise ValueError("""
                    Attempted import of invalid model.
                    Model should result from ProfanityClassifier().train(..)
                    """)
        return model

    def get_model_metrics(self, model):
        """ Extracts and returns model metrics
        """
        metrics = {}
        classes = list(model['gscvScores'].keys())
        metrics['avg_roc_auc'] = model['model_avg_roc_auc']
        for cls in classes:
            metrics['{0}_binary_cv_accuracy'.format(cls)] = model['gscvScores'][cls]['score']
        return metrics

    def __create_binary_training_data(self,
                                      trainPd,
                                      n=1000):
        """ Creates binary training data
        Returns dictionary with, for each class in classes an dataframe df
        where df has two classes of n values. Given class and "other" (True = given, False = other)
        """
        trainPds = {}

        # Retrieve classes from train data
        classes = list(trainPd.keys())
        classes.remove('id')
        classes.remove('comment_text')

        # Create train datasets for each class. Resulting in two types of rows *class and *other
        # Bytheway we're not going to classify anything as "normal".
        #      It results from "other" in all binary classifications
        for cls in classes:
            # Sample "label"(if len(trainPd[cls]) < n, sample w/o replacement. Otherwise with replacement)
            ofLabel = (trainPd[cls] == 1)
            replace = True
            if sum(ofLabel) > n:
                replace = False
            classPd = trainPd[ofLabel].sample(n, replace=replace)
            classPd['label'] = True

            # Sample not "label" #TODO: Stratified sampling? Or an even amount per class?
            # This class will contain LOTS (90%) of "normal" values if we don't do anything about it
            nonClassPd = trainPd[trainPd[cls] != 1].sample(n, replace=False)
            nonClassPd['label'] = False

            # Add as one (vertically joined) dataframe to trainPds
            trainPds[cls] = pd.concat([classPd, nonClassPd], axis=0)

            # We now have, per label, one trainPd with samples of given label and samples of all other classes
            # Note, the True values belong to the label class. But can ALSO belong to other classes.
            #       the False values DON'T belong to the label class. But to any or no amount of other classes
            #       (With same distribution as the original dataset)
        return trainPds

    def __train_on_binary_training_data(self,
                                        binaryTrainingData,
                                        max_dtm_features=100):
        """ Train a classifier for each of the binary classes in binaryTrainingData
        """
        gscvScores = {}       # To hold results
        featuresPds = {}      # To hold dataframes with features
        vectorizers = {}      # To hold vectorizers necessary for testing

        # Get classes and train a model on each class
        classes = list(binaryTrainingData.keys())
        for cls in classes:
            if self.verbose:
                print("Processing class {0}:\n------------------".format(cls))
            currentPd = binaryTrainingData[cls]
            textPd = currentPd[['id', 'comment_text']]
            originalLabelsPd = currentPd[classes]
            classesPd = currentPd['label']

            # Calculate features
            featureDf, vectorizers[cls] = self.__calculate_features(textPd, max_dtm_features)
            featuresPds[cls] = featureDf
            if self.verbose:
                print("\n#Features: {0}".format(featureDf.shape[1]))

            # Parameter optimization
            parameters = {
                'n_estimators': [8, 10, 12],
                'max_depth': [None, 5, 10],
                'max_features': ['auto', 'log2', 0.25, 50]
            }

            # Use random forest
            # grid search on given parameters
            # 10 fold cross validation
            if self.verbose:
                print("Grid search, cross validation..")
            rf = RandomForestClassifier()
            gridSearchCV = GridSearchCV(rf, parameters, cv=10)
            cvs = gridSearchCV.fit(featureDf, classesPd)

            # Best parameters and corresponding CV score
            # When necessary, more results in: cvs.cv_results_
            if self.verbose:
                print("\n--- Results for {0} ---".format(cls))
                print("Best parameters: {0}".format(cvs.best_params_))
                print("With best score: {0}".format(cvs.best_score_))
                print("-------------------------\n")

            # Append scores to history
            gscvScores[cls] = {
                'score': cvs.best_score_,
                'params': cvs.best_params_,
                'max_features': max_dtm_features,
                'best_estimator': cvs.best_estimator_
            }

        return gscvScores, vectorizers

    def __calculate_simple_features(self, message):
        """Cleans message, calculates values on message such as:
        - % of capitals
        - #characters
        - #words
        - #punctuation
        - #!
        - #?

        Returns pandas series of dictionary with features
        """
        return pd.Series({
            'numCapitals': sum(message.count(x) for x in ('Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O',
                                                          'P', 'A', 'S', 'D', 'F', 'G', 'H', 'J', 'K',
                                                          'L', 'Z', 'X', 'C', 'V', 'B', 'N', 'M')),
            'numCharacters': len(message),
            'numWords': len(message.replace('\n', ' ').replace('  ', ' ').split(' ')),
            'numPunctuation': sum(message.count(x) for x in ('!', '@', '#', '$', '%', '^', '&', '*', '(',
                                                             ')', '.', ',', '/', '\\', ']', '[', '{',
                                                             '}', '"', ':', ';', "'", ']', '`', '~', '|')),
            'numExclamation': message.count('!'),
            'numQuestion': message.count('?'),
        })

    def __create_document_term_matrix(self,
                                      messages,
                                      max_features=1000,
                                      strip_accents='unicode',
                                      analyzer='word',
                                      ngram_range=(1, 5),
                                      stop_words='english',
                                      lowercase=True,
                                      max_df=0.9,
                                      min_df=1,
                                      tfidf=False):
        """ Create a TF or TF/IDF matrix
        from given list of messages

        Returns resulting matrix (vectorizer)
        And the dataframe corresponding to this matrix
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
            # Override. Use the tfidf vectorizer
            vectorizer = TfidfVectorizer(max_features=max_features,
                                         strip_accents=strip_accents,
                                         analyzer=analyzer,
                                         ngram_range=ngram_range,
                                         stop_words=stop_words,
                                         lowercase=lowercase,
                                         max_df=max_df,
                                         min_df=min_df)
        vectorizer.fit(messages)

        dtm = vectorizer.transform(messages)
        dtmDf = pd.DataFrame(dtm.toarray(), columns=vectorizer.get_feature_names())

        return dtmDf, vectorizer

    def __calculate_features(self,
                             textPd,
                             max_dtm_features=1000,
                             vectorizers=None):
        """ Calculates features based on text column of dataframe
        Returns features and a dictionary of built vectorizers
        """
        # Simple
        if self.verbose:
            print("Creating simple features..")
        simpleDf = textPd['comment_text'].apply(self.__calculate_simple_features)
        simpleDf = simpleDf.reset_index()
        simpleDf = simpleDf.drop(columns=['index'])

        # Based on vectorizers given or not given, calculate or use
        if vectorizers is None:
            # Vectorizers have not been given. Meaning this is the train step
            #   and we will have to calculate them
            if self.verbose:
                print("No vectorizers given. Creating new on data.")

            # DTM
            if self.verbose:
                print("Creating document term matrix..")
                print("Count..")
            dtmCountDf, dtmCountVectorizer = self.__create_document_term_matrix(list(textPd['comment_text']),
                                                                                max_features=max_dtm_features)

            # TF/IDF
            if self.verbose:
                print("TF/IDF..")
            dtmTfIdfDf, tfIdfVectorizer = self.__create_document_term_matrix(list(textPd['comment_text']),
                                                                             max_features=max_dtm_features,
                                                                             tfidf=True)
        else:
            # Vectorizers have been given. Meaning this is the test step
            #   and we can use the previously trained vectorizers
            if self.verbose:
                print("Vectorizers found. Only applying new data on these vectorizers.")

            # DTM (test)
            if self.verbose:
                print("Count..")
            dtmCount = vectorizers['dtmCountVectorizer'].transform(list(textPd['comment_text']))
            dtmCountDf = pd.DataFrame(dtmCount.toarray(),
                                      columns=vectorizers['dtmCountVectorizer'].get_feature_names())
            dtmCountVectorizer = vectorizers['dtmCountVectorizer']

            # TF/IDF (test)
            if self.verbose:
                print("TF/IDF..")
            dtmTfIdf = vectorizers['tfIdfVectorizer'].transform(list(textPd['comment_text']))
            dtmTfIdfDf = pd.DataFrame(dtmCount.toarray(),
                                      columns=vectorizers['tfIdfVectorizer'].get_feature_names())
            tfIdfVectorizer = vectorizers['tfIdfVectorizer']

        # Merge into one feature DF
        featureDf = pd.concat([simpleDf, dtmCountDf, dtmTfIdfDf], axis=1)
        vectorizers = {'dtmCountVectorizer': dtmCountVectorizer, 'tfIdfVectorizer': tfIdfVectorizer}
        return featureDf, vectorizers

    def __validate_on_remaining_train_data(self,
                                           model,
                                           trainPd,
                                           binaryTrainingData,
                                           sample=5000):
        """ Uses remaining training data
        predicts using trained models
        calculates and shows average area under curve roc result

        Returns average AUC ROC and for each class the AUC ROC
        """
        classes = list(binaryTrainingData.keys())

        # Create a dataframe from the train dataframe
        # without all samples on which the models are trained
        trainedWith = pd.Series()
        for cls in classes:
            trainedWith = pd.concat([trainedWith, binaryTrainingData[cls]['id']], axis=0)
        validatePd = trainPd[~trainPd['id'].isin(trainedWith)]

        # Validate on all train samples which have not been used in training
        if sample is not None:
            samplePd = validatePd.sample(sample)
        else:
            # Use the entire validation Pd
            samplePd = validatePd
        predictedValidationPd = self.predict(model, samplePd)

        # Calculate average area under roc
        print("=============")
        roc_auc = roc_auc_score(samplePd[classes], predictedValidationPd[classes])
        if self.verbose:
            print("Model average roc auc: {0}".format(roc_auc))

        return roc_auc
