import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import re
import string
from contractions import contractions_dict

# import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score
import joblib


def load_data(database_filepath):

    """
    :param database_filepath: path to the database
    :return:
        X(array): training set
        Y(array): test set
        categories(list): label names
    """

    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('DisasterResponseTable', engine)
    X = df['message'].values
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1).values
    categories = df.columns[4:].tolist()

    return X, Y, categories


def expand_contractions(text, contractions_dict):
    """
    :param text(string): message text
    :param contractions_dict(dict): dictionary containing contractions as keys and their expanded forms as values
    :return expanded_text(string): the expanded text
    """

    contractions_pattern = re.compile('({})'.format('|'.join(contractions_dict.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


def expand_match(contraction):
    """
    :param contraction(string): the contraction
    :return expanded_contraction(string) : the expanded contraction
    """

    match = contraction.group(0)
    first_char = match[0]
    expanded_contraction = contractions_dict.get(match) \
        if contractions_dict.get(match) \
        else contractions_dict.get(match.lower())
    expanded_contraction = expanded_contraction
    return expanded_contraction


def tokenize(text):

    """
    :param text(string): a string containing the message
    :return: tokenized_message(list): a list of words containing the processed message
    """

    tokenized_message = []
    try:

        # for unbalanced parenthesis problem
        text = text.replace(')', '')
        text = text.replace('(', '')

        url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

        # get list of all urls using regex
        detected_urls = re.findall(url_regex, text)

        # replace each url in text string with placeholder
        for url in detected_urls:
            text = re.sub(url, "urlplaceholder", text)

        # remove whitespaces
        text = re.sub(r" +", " ", text)
        # expand contractions
        text = expand_contractions(text, contractions_dict)
        # tokenize text
        tokens = word_tokenize(text)
        # initiate lemmatizer
        lemmatizer = WordNetLemmatizer()
        # get stopwords
        stopwords_english = stopwords.words('english')

        for word in tokens:
            # normalize word
            word = word.lower()
            if (word not in stopwords_english and  # remove stopwords
                    word not in string.punctuation):  # remove punctuation

                word = lemmatizer.lemmatize(word)  # lemmatizing word
                tokenized_message.append(word)

    except Exception as e:
        print(e)

    return tokenized_message


def build_model():

    """
    :return: pipeline(object): model for training
    """

    # parameters have been tuned using grid search cv
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, max_df=0.5, max_features=None, ngram_range=(1, 2))),
        ('tfidf', TfidfTransformer(use_idf=False)),
        ('clf', MultiOutputClassifier(
            XGBClassifier(colsample_bytree=0.7, gamma=0.4, learning_rate=0.25, max_depth=10,
                                      min_child_weight=7)))
    ])
    return pipeline


def grid_search_cv(pipeline, X_train, y_train, params_filepath):

    """
    :param pipeline(object): pipeline for data processing and training
    :param X_train(array): training set features
    :param y_train(array): training set labels
    :param params_filepath(string): path to store the best parameters
    :return(dict) : best parameters for training
    """

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000),
        'tfidf__use_idf': (True, False),
        'clf__estimator__learning_rate': [0.05, 0.15, 0.25],  # shrinks feature values for better boosting
        'clf__estimator__max_depth': [4, 6, 8, 10],
        'clf__estimator__min_child_weight': [1, 3, 5, 7],   # sum of child weights for further partitioning
        'clf__estimator__gamma': [0.0, 0.1, 0.2, 0.3, 0.4],  # prevents overfitting, split leaf node if min. gamma loss
        'clf__estimator__colsample_bytree': [0.3, 0.4, 0.5, 0.7]  # subsample ratio of columns when tree is constructed
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=10, verbose=10)

    cv.fit(X_train, y_train)
    joblib.dump(cv, params_filepath)

    return cv.best_params_


def evaluate_model(model, X_test, Y_test, category_names):
    """
    :param model(object): trained model
    :param X_test(array): test set data
    :param Y_test(array): test set target labels
    :param category_names(list): label names
    :return: None
    """

    y_pred = model.predict(X_test)
    report = {}
    f1_scores = []
    for idx in range(y_pred.shape[1]):
        report[category_names[idx]] = classification_report(Y_test[:, idx], y_pred[:, idx], output_dict=True)

    for idx, col in enumerate(y_pred.T):
        f1_scores.append(f1_score(Y_test.T[idx], col, average='weighted'))
    print("Mean f1-score: {}".format(np.mean(f1_scores)))


def save_model(model, model_filepath):
    """
    :param model(object): trained model
    :param model_filepath(string): path to saved model
    :return: None
    """
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=4)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
