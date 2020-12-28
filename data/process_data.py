import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    :param messages_filepath: path of messages.csv file
    :param categories_filepath: path of categories.csv file
    :return df(dataframe): merged dataframe containing data from both files
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on=['id'])

    return df


def clean_data(df):
    """
    :param df: merged dataset that needs to be cleaned
    :return: cleaned dataframe
    """

    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.values[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = [cat[:-2] for cat in row]

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # concatenate the original dataframe with the new `categories` dataframe
    # drop the original categories column from `df`
    df = pd.concat([df, categories], axis=1)
    df.drop(['categories'], axis=1, inplace=True)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    # convert values to 0's and 1's
    df = df.replace(2, 1)

    # drop column child_alone since it contains 0's only
    df.drop(['child_alone'], axis=1, inplace=True)

    return df


def save_data(df, database_filename):
    """
    :param df: cleaned dataset in a dataframe
    :param database_filename: name of the database where the data will be stored
    :return: None
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('DisasterResponseTable', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()