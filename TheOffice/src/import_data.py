import pandas as pd
import numpy as np
import warnings
import string
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler

#characters used in the character model
main_characters = ["Michael", "Dwight", "Jim", "Pam", "Ryan", "Andy", "Robert", "Stanley", "Kevin", "Meredith", "Angela", "Oscar", "Roy", "Phillis",
"Jan", "Kelly", "Toby", "Creed", "Darryl", "Erin", "Gabe", "Holly", "Nellie", "Clark", "Pete"]

#TARGETS
IMDB_scores = [7.5, 8.3, 7.9, 8.1, 8.4, 7.8,
        8.8, 8.2, 8.4, 8.4, 8.2, 8.2, 8.6, 8.2, 8.4, 8.9, 8.6, 9.1, 8.3, 7.9, 8.2, 8.3, 8.5, 8.3, 8.1, 8.4, 8.7, 9.4,
        9.0, 8.2, 8.6, 8.0, 8.2, 8.0, 8.6, 8.8, 8.3, 8.8, 8.5, 8.7, 8.8, 8.2, 8.2, 8.9, 8.5, 9.0, 8.8, 8.8, 8.8, 9.2, 9.4,
        8.8, 8.3, 8.5, 8.7, 8.8, 8.5, 8.2, 8.7, 9.4, 8.1, 8.7, 8.4, 7.9, 9.3,
        8.8, 8.3, 8.0, 8.1, 8.1, 8.5, 8.2, 8.6, 8.8, 8.4, 8.7, 8.0, 9.7, 8.2, 8.2, 8.0, 8.7, 8.3, 8.4, 8.3, 8.7, 8.7, 9.2, 8.2, 8.7, 9.0,
        8.8, 8.1, 8.0, 9.4, 9.4, 7.6, 8.6, 8.2, 8.1, 8.6, 8.2, 8.3, 8.5, 6.8, 7.7, 8.1, 8.4, 8.5, 7.7, 7.8, 8.6, 7.9, 8.0, 8.2, 7.8, 8.0,
        8.4, 8.3, 8.2, 7.9, 7.8, 8.2, 7.4, 7.9, 7.7, 8.2, 9.0, 8.3, 7.7, 8.5, 8.4, 9.4, 7.5, 9.3, 7.8, 9.0, 9.8, 7.7, 8.7, 8.8,
        8.2, 8.1, 7.3, 8.1, 7.6, 7.7, 7.7, 6.9, 7.7, 8.0, 7.9, 8.0, 7.5, 7.7, 7.8, 8.1, 7.8, 7.8, 6.7, 7.1, 7.1, 7.1, 7.7, 7.7,
        7.7, 7.2, 7.4, 7.8, 7.0, 7.7, 7.6, 7.8, 8.3, 7.6, 7.9, 8.0, 7.6, 7.5, 7.4, 8.2, 7.5, 8.0, 8.0, 8.0, 9.0, 9.5, 9.8]

#Creates a file for target data
def make_targets():
    scores = np.array(IMDB_scores)
    np.save("../data/IMDB_scores", scores)

#splits data and saves as numpy arrays for easier execution
def preprocess(filename, output_name):

    data = pd.read_csv(filename)

    x_train, x_test, y_train, y_test = train_test_split(data, IMDB_scores, \
                                                        test_size = 0.2)

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)                               

    np.save(output_name + "_x_train", x_train)
    np.save(output_name + "_x_test", x_test)
    np.save(output_name + "_y_train", y_train)
    np.save(output_name + "_y_test", y_test)

def make_csv():
    """
    Creates a CSV file from The Office data set where each word is a feature

    Parameters: none

    Returns:    none
    """
    data = pd.read_csv("../data/the-office-lines-scripts.csv")
    warnings.simplefilter(action='ignore', category=FutureWarning)
    data = data[data.deleted == False]
    data = data.drop(['id', 'deleted', 'scene'], axis = 1)

    clean_data = pd.DataFrame()

    season = 0
    episode = 0
    cumulative_episode = -1
    data_top = data.head()

    for index, row in data.iterrows():
        if row['season'] != season:
            season = row['season']
        if row['episode'] != episode:
            cumulative_episode += 1
            episode = row['episode']
            clean_data = clean_data.append({'_cumulative_episode': cumulative_episode, '_season': season, '_episode': episode}, ignore_index = True, sort = False)
        word_dict, line_length = make_dictionary(row['line_text'])
        word_dict["_" + row['speaker'].replace(" ", "")] = line_length
        clean_data = clean_data.fillna(0)
        for key, value in word_dict.items():
            if key not in clean_data.columns:
                clean_data[key] = 0
            if clean_data.at[cumulative_episode, key] == np.nan:
                clean_data.at[cumulative_episode, key] = 0
            clean_data.at[cumulative_episode, key] += value


    clean_data = clean_data.fillna(0)

    #To delete common words
    clean_data = delete_common_words(clean_data)

    #alter based on if deleting common words
    clean_data.to_csv(r'../data/all_words.csv')



def make_character_csv(use_lines):
    """
    Creates a CSV file from original The Office data set where the features are
    the number of words spoken or number of lines spoken by each character in each episode

    Parameters: use_lines - A boolean variable. If True, features are the number
                            of lines spoken. If False, features are the number
                            of words spoken.

    Returns:    none
    """
    data = pd.read_csv("../data/the-office-lines-scripts.csv")
    warnings.simplefilter(action='ignore', category=FutureWarning)
    data = data[data.deleted == False]
    data = data.drop(['id', 'deleted', 'scene'], axis = 1)

    clean_data = pd.DataFrame()

    season = 0
    episode = 0
    cumulative_episode = -1
    data_top = data.head()

    for index, row in data.iterrows():
        if row['season'] != season:
            season = row['season']
        if row['episode'] != episode:
            cumulative_episode += 1
            episode = row['episode']
            clean_data = clean_data.append({'_cumulative_episode': cumulative_episode, '_season': season, '_episode': episode}, ignore_index = True, sort = False)
        line = row['line_text']

        twss_count = count_twss(line)
        if "twss_count" not in clean_data.columns:
                clean_data["twss_count"] = 0
        clean_data.at[cumulative_episode, "twss_count"] += twss_count

        if row['episode'] == 1:
            if "first_episode" not in clean_data.columns:
                clean_data["first_episode"] = 0
            clean_data.at[cumulative_episode, "first_episode"] = 1

        if (row['season'] == 1 and row['episode'] == 6) or \
        (row['season'] == 2 and row['episode'] == 22) or \
        (row['season'] == 3 and row['episode'] == 23) or \
        (row['season'] == 4 and row['episode'] == 14) or \
        (row['season'] == 5 and row['episode'] == 26) or \
        (row['season'] == 6 and row['episode'] == 26) or \
        (row['season'] == 7 and row['episode'] == 24) or \
        (row['season'] == 8 and row['episode'] == 24) or \
        (row['season'] == 9 and row['episode'] == 23):
            if "last_episode" not in clean_data.columns:
                clean_data["last_episode"] = 0
            clean_data.at[cumulative_episode, "last_episode"] = 1

        text = line.split()
        length = len(text)
        key = row['speaker']
        clean_data = clean_data.fillna(0)
        if key in main_characters:
            if key not in clean_data.columns:
                clean_data[key] = 0
            if clean_data.at[cumulative_episode, key] == np.nan:
                clean_data.at[cumulative_episode, key] = 0
            if use_lines:
            #counts number of words spoken per character
                clean_data.at[cumulative_episode, key] += length
            #counts number of lines spoken per character
            else:
                clean_data.at[cumulative_episode, key] += 1
        clean_data = clean_data.fillna(0)

    #alter based on if deleting common words
    if use_lines:
        clean_data.to_csv(r'../data/character_lines_twss.csv')
    else:
        clean_data.to_csv(r'../data/character_words_twss.csv')


def count_twss(line):
    """
    Counts the number of times 'thats what she said' is said in a line

    Parameters:
        line - line from the script

    Returns:
        count of times said
    """
    text = line.split()
    table = str.maketrans('', '', string.punctuation)
    stripped = [t.translate(table) for t in text]
    stripped = [i.lower() for i in stripped]
    twss_count = 0
    for i in range(len(stripped)):
        if stripped[i] == "thats" and (i + 4) <= len(stripped):
                if stripped[i + 1] == "what" and stripped[i + 2] == "she" and stripped[i + 3] == "said":
                    twss_count += 1
    return twss_count

def make_dictionary(line):
     """
    Makes a word dictionary with the number of occurrences of every word in the line

    Parameters:
        line - line from the script

    Returns:
        dictionary with word counts
    """
    dictionary = {}
    text = line.split()
    length = len(text)
    table = str.maketrans('', '', string.punctuation)
    stripped = [t.translate(table) for t in text]
    stripped = [i.lower() for i in stripped]
    for i in range(len(stripped)):
        if stripped[i] in dictionary:
            dictionary[stripped[i]] += 1
        else:
            dictionary[stripped[i]] = 1
    return dictionary, length

def delete_common_words(data):
     """
    Deletes common words from a given data frame

    Parameters:
        data - dataframe

    Returns:
        dataframe with common words deleted
    """
    file = open(r"../data/commonwords.txt","r")
    words = file.read()
    file.close()
    common_words = np.array(words.split(','))
    for (columnName, columnData) in data.iteritems():
        if (columnName in common_words):
            data.drop(columns = [columnName], inplace = True)
    return data

def main():
    # make_csv()
    # make_character_csv(True)
    # make_character_csv(False)
    # preprocess('../data/all_words.csv', '../data/all_words')
    # preprocess('../data/character_lines.csv', '../data/character_lines_twss')
    # preprocess('../data/character_words.csv', '../data/character_words_twss')
    make_targets()


if __name__ == "__main__":
    main()
