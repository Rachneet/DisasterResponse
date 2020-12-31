import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify

import joblib
from sqlalchemy import create_engine

import plotly_express as px
from plotly.graph_objects import Bar

from sklearn.feature_extraction.text import CountVectorizer
from models.train_classifier import tokenize

app = Flask(__name__)
app.static_folder = 'static'

# def tokenize(text):
#     tokens = word_tokenize(text)
#     lemmatizer = WordNetLemmatizer()
#
#     clean_tokens = []
#     for tok in tokens:
#         clean_tok = lemmatizer.lemmatize(tok).lower().strip()
#         clean_tokens.append(clean_tok)
#
#     return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponseTable', engine)

# load model
model = joblib.load("../models/classifier.pkl")


def get_top_n_words(corpus, n=None):

    """
    :param corpus(string): message text
    :param n(int): top frequent words
    :return(list): most frequent words and their counts
    """

    vec = CountVectorizer(tokenizer=tokenize).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in
                  vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1],
                        reverse=True)
    return words_freq[:n]


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # get category counts
    # take second element for sort
    def takeSecond(elem):
        return elem[1]

    vals = sorted([(col, df[col].value_counts()[1]) for col in df.columns[4:].tolist()], reverse=True, key=takeSecond)
    cat_counts = [x[1] for x in vals]
    cat_names = [' '.join(x[0].split('_')) for x in vals]

    # most frequent words in the user messages
    # Convert most freq words to dataframe for plotting bar plot
    top_words = get_top_n_words(df.message, n=22)
    top_words = [i for i in top_words if not (i[0] == 'u' or i[0] == '..')]
    top_df = pd.DataFrame(top_words)
    top_df.columns = ["Word", "Freq"]
    colors = px.colors.qualitative.Dark24

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

        {
            'data': [
                Bar(
                    x=cat_names,
                    y=cat_counts,
                    marker_color='#66c56c'
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': -45,
                    'automargin': True
                }
            }
        },

        {
        'data': [
                    Bar(
                        x=top_df.Word.values,
                        y=top_df.Freq.values,
                        marker_color=colors

                    )
                ],

                'layout': {
        'title': 'Most Frequent Words in the Corpus',
        'yaxis': {
            'title': "Count"
        },
        'xaxis': {
            'title': "Word",
            'tickangle': -45,
            'automargin': True
        }
    }
        }
    ]


    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='127.0.0.1', port=5000, debug=True)


if __name__ == '__main__':
    main()