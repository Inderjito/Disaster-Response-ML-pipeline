import sys

import pandas as pd
from sqlalchemy import create_engine
import nltk
import joblib 
import re
import time

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download(['punkt', 'wordnet','stopwords'])

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


import nltk
nltk.download(['punkt', 'wordnet'])
nltk.download('averaged_perceptron_tagger')
pd.set_option('display.max_colwidth', -1)


def load_data(database_filepath):
    """
    Function:
       Load clean dataset from Sqlite database
       Args:
       database_filepath: the path of sqlite database
       Return:
        
         Message data
         target dataframe
         target labels list
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('messages_disaster_final', engine)
    X = df['message']  # Message Column
    Y = df.drop(['id','message','original','genre'],axis=1)
    category_names = df.columns[4:]
    return X, Y, category_names
    
    pass
   
       

def tokenize(text):
   """    
   This function normalize text, remove punctuation and stop words
   Args:
        Text tokenize(str)
        
   Returns:
           words to orginal form
        
    """     
   text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()) 
   tokens = word_tokenize(text)
   stop_words = stopwords.words("english")
   words = [w for w in tokens if w not in stop_words]
   stemmer = PorterStemmer()
   stemmed = [stemmer.stem(w) for w in words] 
   lemmatizer = WordNetLemmatizer()
   lemmed = [lemmatizer.lemmatize(w) for w in stemmed]
    
   return lemmed

    

def build_model():
    """
    Returns the GridSearchCV model
    
    Args:
        None
    Returns:
        cv: Grid search model object
    """
    
    # define the step of pipeline
    pipeline = Pipeline([
    ('vect',CountVectorizer(tokenizer=tokenize,ngram_range=(1,2),max_df=0.75)),
    ('tfidf', TfidfTransformer()),
    ('clf',MultiOutputClassifier(RandomForestClassifier(n_jobs=-1)))])

    # define the parameters to fine tuning
    parameters = {
        'vect__max_features': (None,10000,30000)
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    This function evaluates model
    
    INPUT model, X_test, Y_test, category_names):
    OUTPUT: provide multi target accuracy and classification report metrics
    """
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))

def save_model(model, model_filepath='random_forest.pkl'):
    """
    Save model as pickle file
    
    Args: 
        model (sklearn estimator): model
        model_filepath (str): filepath to save model
    Return:
        None
    """
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()