_Note: A lot of the following is a copy paste from the explanation within the ipynb file_

# Profanity Classification
Based on [this kaggle challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) called "Toxic Comment Classification Challenge - Identify and classify toxic online comments"

### Usage
#### Install necessary packages
Make sure you have python3 and pip(3) running:
- [python3](https://www.digitalocean.com/community/tutorials/how-to-install-python-3-and-set-up-a-local-programming-environment-on-macos)
- [pip3](http://itsevans.com/install-pip-osx/)

Use pip3 ([on virtualenv when you can](https://docs.python.org/3/library/venv.html)) to install:
- pandas
- numpy
- sklearn 
- matplotlib

#### Download the datasets
Download the datasets train.csv and test.csv from [the Kaggle page](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) (after required login, acceptance of agreements)
- You can also download model.pickle from this git and use `load_model` which is explained later.
- Note: In the examples, all files are in the same directory as the code

#### Python usage
```
from ProfanityClassifier import ProfanityClassifier
import pandas as pd
```

Initiate the class
```
pcf = ProfanityClassifier(verbose=False)
```

Train model on train data
Optional: `numTrainSamplesPerClass (default: 1000)`, `maxFeatures (default: 250)`
```
trainedModel = pcf.train("train.csv")
```

OR optionally load existing model:
```
trainedModel = pcf.load_model("model.pickle")
```

Look into model metrics (A great method to compare an old to a new model):
```
pcf.get_model_metrics(trainedModel)
```

Predict samples on model.
Possibilities:

1. Samples are given in a csv (with rows of format: id, text\n)
- Returns dictionary or DataFrame depending on "dictOut"
```
predictions = pcf.predict_on_csv(trainedModel, "test.csv", dictOut=True)
```

2. Samples are given in a dictionary or list of dictionaries
- Returns dictionary or DataFrame depending on "dictOut"
```
predictions = pcf.predict_on_dictionary(trainedModel, {
    'id':'1234',
    'comment_text':'Hi, this is a comment :)'
}, dictOut=True)
```

3. Samples are given in a pandas DataFrame:
```
predictions = pcf.predict(trainedModel, pd.DataFrame([{
    'id':'1234',
    'comment_text':'Hi, this is a comment :)'
}]))
```

And optionally save model:
```
pcf.save_model(trainedModel, "model.pickle")
```

### Example usage with existing model
```
from ProfanityClassifier import ProfanityClassifier
import pandas as pd

pcf = ProfanityClassifier(verbose=False)

# Import model
trainedModel = pcf.load_model("model.pickle")

# Predict on model
predictions = pcf.predict_on_dictionary(trainedModel, {
    'id':'1234',
    'comment_text':'Hi, this is a comment :)'
}, dictOut=True)
```

### Example re-training of model
```
from ProfanityClassifier import ProfanityClassifier
import pandas as pd

pcf = ProfanityClassifier(verbose=False)

# Train model
trainedModel = pcf.train("train.csv")

# Show metrics
pcf.get_model_metrics(trainedModel)

# Save model
pcf.save_model(trainedModel, "model.pickle")
```

----------------------------------

### Process:
The process can be found below. 
1. First some shallow reading of existing research
2. Analysis of the data. What do we have?
3. Interpretation of the data
4. Setting out our options of what to do
5. Selecting option to build (multiple are possible)
6. Explanation of what is built
7. Results
- _Note: Here we could/should re-evaluate, find out how to improve. Basically starting at 3/4_
8. Conclusion

### Quick research
Turned up these papers (pdf warning and such):
1. [Text classifcation of short messages. Lundborg et al.](http://lup.lub.lu.se/luur/download?func=downloadFile&recordOId=8928009&fileOId=8928011)
2. [Automatic Classification of Abusive Language and Personal Attacks in Various Forms of Online Communication. Bourgonje et al.](https://link.springer.com/content/pdf/10.1007%2F978-3-319-73706-5_15.pdf)
3. [Automatic Detection of Cyberbullying in Social Media Text. van Hee et al.](https://arxiv.org/pdf/1801.05617.pdf) and a [different version](https://biblio.ugent.be/publication/6969774/file/6969839.pdf)
4. [Harnessing the Power of Text Mining for the Detection of Abusive Content in Social Media. Chen et al.](https://arrow.dit.ie/cgi/viewcontent.cgi?referer=https://scholar.google.nl/scholar?as_ylo=2014&q=profanity+text+classification&hl=en&as_sdt=0,5&httpsredir=1&article=1196&context=scschcomcon)

#### A  shallow scan of these papers/thesis:
1. They are also looking into a multi class problem. Although generally less than our 7 classes.
2. They, among English, are working on Swedish, Dutch (well, Belgium ;)) data.
3. Annotation seems to be one of the hardest problems. Already done for us :D
4. There could be a nice scientific comparison to peer work. If time
5. n-grams (character, word, skip-grams)
   - 1-3 word n-grams and 1-6 character n-grams
   - skipgrams using NER from spacy, mcparseface or NLTK. For example 2-grams in the list of nouns 
6. (Feature) Spelchecker results
7. (Feature) Sentiment analysis (In this case: #positive words/#total words, #negative words/#total words)
8. (Feature) Linguistic features
   - #words, #characters
   - #uppercase words (normalized)
   - #uppercase characters (normalized) (-> avg ratio uppercase characters per word?)
   - Longest word
   - Average word length
   - #one letter tokens, #number of one letter tokens / #words.
   - #punctuation, spaces, exclamation marks, question marks, at signs and commas
9. (Feature) You can extract sytactic features from word trees (generated by for example spacy, I remember).
   - Word + parent
   - Word + grandparent
   - Word + children
   - Word + siblings of parent
10. (Feature) Specifically engineered term lists
   - Binary features indicating that a term from given list is in the text. Lists are researcher engineered and contain for example words indicating bullying (in the case of paper 3)
11. (Classifiers Used) Naive Bayes, SVM, Random Forest, Neural Networks

## Some stats on the data
Train data:
   - #rows: 159571
   - columns: ['id', 'comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

Test data:
   - #rows: 153164
   - columns: ['id', 'comment_text']

Can one line have multiple true labels? (train)
   - #rows with more than one true label: 9865

How many of each label is there?
   - #toxic:  15294 (=9.58% of train data)
   - #severe_toxic: 1595  (=1.0% of train data)
   - #obscene:   8449  (=5.29% of train data)
   - #threat: 478   (=0.3% of train data)
   - #insult: 7877  (=4.94% of train data)
   - #identity_hate:   1405  (=0.88% of train data)
   - #normal: 143346   (=89.8321% of train data)

Train:
   - min:  6
   - max:  5000
   - median:  205.0
   - mean: 394.07
   - stddev:  590.72
   - 10th perc:  47.0
   - 90th perc:  889.0

Test:
   - min:  1
   - max:  5000
   - median:  180.0
   - mean: 364.88
   - stddev:  592.49
   - 10th perc:  38.0
   - 90th perc:  804.0

Train toxic == 1:
   - min:  8
   - max:  5000
   - median:  123.0
   - mean: 295.25
   - stddev:  617.36
   - 10th perc:  34.0
   - 90th perc:  582.0

Train severe_toxic == 1:
   - min:  8
   - max:  5000
   - median:  94.0
   - mean: 453.64
   - stddev:  1090.65
   - 10th perc:  31.0
   - 90th perc:  891.0

Train obscene == 1:
   - min:  8
   - max:  5000
   - median:  110.0
   - mean: 286.78
   - stddev:  641.05
   - 10th perc:  32.0
   - 90th perc:  544.0

Train threat == 1:
   - min:  19
   - max:  5000
   - median:  121.0
   - mean: 307.74
   - stddev:  729.44
   - 10th perc:  42.0
   - 90th perc:  499.6

Train insult == 1:
   - min:  8
   - max:  5000
   - median:  112.0
   - mean: 277.28
   - stddev:  622.51
   - 10th perc:  33.0
   - 90th perc:  520.4

Train identity_hate == 1:
   - min:  18
   - max:  5000
   - median:  114.0
   - mean: 308.54
   - stddev:  691.38
   - 10th perc:  32.0
   - 90th perc:  616.6

### Interpretation
- Multilabel classification problem. One row can have more than one label
- Skewed dataset. Lots (90%) of "Normal", only 0.88% of the rows has the label "threat" 
- 90% is of normal class
- Lots of rows have more than one label. For example. Of the total, 9.58% in the train data is "toxic". Which is almost 100% of the non normal train rows.

### What to do?
We could (among others):
1. Bootstrap our samples in the underrepresented classes
2. Undersample the overrepresented classes (Or in combination with 1.)
3. Do nothing, let the classification algorithm figure it out (For example Random Forest can handle some skewness)
4. Do a binary classification for each of 8 classes (Is it normal or other, is it obscene or other. etc. We can now sample non skewed sets per class)
5. Create, from the 7 binary columns one integer (0-127). 

### What should we do?
- I think, 4. Because 1, 2 are hard due to the multilabel properties. Sampling more from the "threat" class will also result lots of samples from other classes. We would still have a skewed set. (it is solvable though). Also, this could result in a biased train set.
- 3 is also not optimal because it is not slightly skewed. We have a significant skewness. 
- 4 allows us to create equal sets and still get a probability per class. Which is what the Kaggle challenge asks for.
- 5: I for now wouldn't immediately know how to handle the results. We would have to map back from this continuous value to 7 probabilities.

_(So, I'm going with 4 and not trying more due to time constraints)_

## Implementation

1. create_binary_training_data
   - Creates, for each class a training set with n True and n False samples
2. train_on_binary_training_data 
   - For each class separately calculate features and train an model
   - Creates some simple counting features
   - Uses create_document_term_matrix to create TF and TF/IDF matrices
   - Trains a Random Forest Classifier using Grid Search and 10 fold cross validation
3. show_feature_importances
   - Shows for each of the models the n most important features
   - Allows for visual inspection. Does it make sense? Does it look like we could improve on this? 
4. validate_on_remaining_train_data
   - In train_on_binary_training_data, only part of the training data has been sampled to be used
   - Use the rest of this training data as an validation set
   - predict on these validation samples using the trained models (in "predict")
   - Calculate the avg ROC AUC on the results. Kaggle ranks on this value (on the test set that is)
5. predict
   - Uses the existing models to predict for each class the probability that this comment falls in given class
6. save_predictions
   - Saves the comment_id's with the predicted probabilities to an CSV (to be uploaded to Kaggle)
7. print_random_classifications
   - For visual inspection
   - Shows comment_text for n random samples with actual label and predicted labels

## Results
- Train Cross Validation results are in the 0.8-0.95 range (for all binary classes)
- Validation avg ROC AUC is ~0.9 (Note: The validation set has different ratios than the train set because of the sampling (and removel of the train data)
- Visual inspection shows some correctness

## Conclusion
There are still lots of possibilities for improvement. 
A lot of these have been mentioned in the introduction above.
In addition, features from topics resulting from LDA might improve on our results.

They are not implemented due to time constraints on my part. 
Also, for example spell checking and NER tools will take significant processing time. 
When distributed, this should not be a problem but on my 2 core mac it is ;)

In my opinion, the numbers look good. It is fairly accurate (although 10% False classification will result in 15k misclassifications).

Visual inspection of the results show some correctness although I find it hard to believe it is near good enough to use. Also it doesn't necessarily seem to be close to numbers resulting from the CV and separate validation. I wouldn't be surprised if something is wronge here somewhere. (And am looking for it ;). It should have to do with the skewness, binary sampling and resulting validation set distributions)