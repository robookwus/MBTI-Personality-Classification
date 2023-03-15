import pandas as pd
from langdetect import detect
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from collections import defaultdict


def load_comments_for_respective_class(class_name):
    path_to_feather_file = "samples/disprop_sample100k_total.feather"
    df = pd.read_feather(path_to_feather_file)

    dataset = df.loc[df["labels"] == class_name, "comments"]

    return dataset


def get_lang_distribution(dataset):
    language_counts = defaultdict(int)

    for x in range(dataset.shape[0]):
        try:
            language = detect(dataset[x])
        except Exception:
            print("Unclassified Language detected!")
            language = "uc"

        language_counts[language] += 1

    return language_counts


def draw_distribution(dataset, class_title):

    # Erstelle eine Tabelle mit Sprachkürzeln und vollständigen Namen
    language_names = {'en': 'English', 'da': 'Danish', 'cs': 'Czech', 'sq': 'Albanian', 'pl': 'Polish',
                      'it': 'Italian', 'so': 'Somali', 'no': 'Norwegian', 'es': 'Spanish', 'fr': 'French',
                      'af': 'Afrikaans', 'de': 'German', 'ca': 'Catalan', 'id': 'Indonesian', 'et': 'Estonian',
                      'cy': 'Welsh', 'pt': 'Portuguese', 'tl': 'Tagalog', 'sv': 'Swedish', 'ro': 'Romanian',
                      'he': 'Hebrew', 'hr': 'Croatian', 'tr': 'Turkish', 'nl': 'Dutch', 'sk': 'Slovak',
                      'mk': 'Macedonian', 'ru': 'Russian', 'lt': 'Lithuanian',
                      'zh-cn': 'Chinese', 'hu': 'Hungarian', 'sw': 'Swahili', 'sl': 'Slovenian', 'uc': 'unclassified'}

    # Gehe durch das Dictionary und ändere die Schlüssel zu den vollständigen Sprachnamen
    language_counts_full_names = {}
    for key, value in dataset.items():
        try:
            language_counts_full_names[language_names[key]] = value
        except KeyError:
            print("KeyError")
            pass

    sorted_language_counts = dict(sorted(language_counts_full_names.items(), key=lambda x: x[1], reverse=True))

    # Extrahiere die Sprachnamen und die Anzahl der Kommentare als separate Listen
    languages = [language for language in sorted_language_counts.keys()]
    counts = [count for count in sorted_language_counts.values()]

    en_count = counts[0]
    other_counts = 0
    for x in range(len(counts)-1):
        other_counts += counts[x+1]
    total_counts = 0
    for x in range(len(counts)):
        total_counts += counts[x]

    number_of_languages = len(sorted_language_counts)
    en_percentage = round(float(100*(en_count/total_counts)), 2)

    # Erstelle ein Balkendiagramm
    plt.figure(figsize=(10, 6))
    plt.bar(languages, counts)
    plt.xticks(rotation=90)
    plt.xlabel('Respective Language')
    plt.ylabel('Number of Comments')
    plt.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.2)
    plt.title('Distribution of comments per language for class ' + class_title, fontsize=13)

    plt.text(5, 6000, 'considered comments: ' + str(total_counts),
             ha='center', va='center', fontsize=12)
    plt.text(5, 5600, 'detected languages: ' + str(number_of_languages),
             ha='center', va='center', fontsize=12)
    plt.text(15, 5600, 'proportion of english comments: ' + str(en_percentage) + "%",
             ha='center', va='center', fontsize=12)

    for language, count in sorted_language_counts.items():
        plt.text(language, count, str(count), ha='center', fontsize=12)

    figure_name = "Language_Distribution_for_class_" + class_title + ".png"
    plt.savefig("figures/" + figure_name)

    print("Language distribution figure saved.")


def bow(dataset_1, dataset_2, class_title_1, class_title_2):
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    stemmer = SnowballStemmer('english')

    # collect en comments for class 1
    en_dataset_1 = []
    en_dataset_2 = []

    for x in range(dataset_1.shape[0]):
        try:
            language = detect(dataset_1[x])
            if language == "en":
                en_dataset_1.append(dataset_1[x])
        except Exception:
            pass

    # collect en comments for class 2
    for x in range(dataset_2.shape[0]):
        try:
            language = detect(dataset_2[x])
            if language == "en":
                en_dataset_2.append(dataset_2[x])
        except Exception:
            pass

    # preprocessing

    def preprocessing(words_list):

        bow_dictionary = defaultdict(int)

        for word in words_list:
            if "." or "," or ";" or ":" or "!" or "?" in word:
                word = word.replace(".", "")
                word = word.replace(",", "")
                word = word.replace(":", "")
                word = word.replace(";", "")
                word = word.replace("!", "")
                word = word.replace("?", "")
            if "0" or "1" or "2" or "3" or "4" or "5" or "6" or "7" or "8" or "9" in word:
                word = word.replace("0", "")
                word = word.replace("1", "")
                word = word.replace("2", "")
                word = word.replace("3", "")
                word = word.replace("4", "")
                word = word.replace("5", "")
                word = word.replace("6", "")
                word = word.replace("7", "")
                word = word.replace("8", "")
                word = word.replace("9", "")
            if "*" or "+" or "-" or "/" or "%" or "(" or ")" or "[" or "]" in word:
                word = word.replace("*", "")
                word = word.replace("+", "")
                word = word.replace("-", "")
                word = word.replace("/", "")
                word = word.replace("%", "")
                word = word.replace("(", "")
                word = word.replace(")", "")
                word = word.replace("[", "")
                word = word.replace("]", "")
            if "i´ve" or "you´ve" or "we`ve" or "i´d" in word:
                word = word.replace("i´ve", "")
                word = word.replace("you´ve", "")
                word = word.replace("we´ve", "")
                word = word.replace("i´d", "")
            if "http" in word:
                word = ""
            word = stemmer.stem(word)
            if word != "" and word not in stop_words:
                bow_dictionary[word] += 1
        return bow_dictionary

    # creating bow for class 1 ------------------------------------------------

    general_words = []

    for x in range(len(en_dataset_1)):
        comment = en_dataset_1[x]
        comment = comment.lower()
        general_words += comment.split(" ")

    sorted_bow = dict(sorted(preprocessing(general_words).items(), key=lambda x: x[1], reverse=True))

    # Extrahiere die Wörter und die Anzahl als separate Listen
    words_1 = [word for word in sorted_bow.keys()]
    counts_1 = [count for count in sorted_bow.values()]

    # creating bow for class 2 ------------------------------------------------

    general_words = []

    for x in range(len(en_dataset_2)):
        comment = en_dataset_2[x]
        comment = comment.lower()
        general_words += comment.split(" ")

    bow_dictionary = preprocessing(general_words)
    sorted_bow = dict(sorted(bow_dictionary.items(), key=lambda x: x[1], reverse=True))

    # Extrahiere die Wörter und die Anzahl als separate Listen
    words_2 = [word for word in sorted_bow.keys()]
    counts_2 = [count for count in sorted_bow.values()]

    # deleting words that occur in both datasets and shortening to 50 tokens

    for word in words_1:
        if word in words_2:
            words_1.remove(word)
            words_2.remove(word)

    words_1 = words_1[0:50]
    counts_1 = counts_1[0:50]

    words_2 = words_2[0:50]
    counts_2 = counts_2[0:50]

    # plotting bow 1

    # Erstelle ein Balkendiagramm
    plt.figure(figsize=(10, 6))
    plt.bar(words_1, counts_1)
    plt.xticks(rotation=90)
    plt.xlabel('Respective Token (up to 50)')
    plt.ylabel('Number of Tokens in BoW')
    plt.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.2)
    plt.title('Bag of words representation for class ' + class_title_1)

    figure_name = "BoW_class_" + class_title_1 + ".png"
    plt.savefig("figures/" + figure_name)

    print("BoW figure for class 1 saved.")

    # plotting bow 2

    # Erstelle ein Balkendiagramm
    plt.figure(figsize=(10, 6))
    plt.bar(words_2, counts_2)
    plt.xticks(rotation=90)
    plt.xlabel('Respective Token (up to 50)')
    plt.ylabel('Number of Tokens in BoW')
    plt.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.2)
    plt.title('Bag of words representation for class ' + class_title_2)

    figure_name = "BoW_class_" + class_title_2 + ".png"
    plt.savefig("figures/" + figure_name)

    print("BoW figure for class 2 saved.")


def binary_compare_of_datasets(name_of_dataset_1, name_of_dataset_2,
                               create_language_distribution, create_bag_of_words):

    dataset_1 = load_comments_for_respective_class(class_name=name_of_dataset_1)
    dataset_2 = load_comments_for_respective_class(class_name=name_of_dataset_2)

    if create_language_distribution:
        dataset_1_langs = get_lang_distribution(dataset=dataset_1)
        dataset_2_langs = get_lang_distribution(dataset=dataset_2)
        draw_distribution(dataset=dataset_1_langs, class_title=name_of_dataset_1)
        draw_distribution(dataset=dataset_2_langs, class_title=name_of_dataset_2)

    if create_bag_of_words:
        bow(dataset_1=dataset_1, dataset_2=dataset_2,
            class_title_1=name_of_dataset_1, class_title_2=name_of_dataset_2)


binary_compare_of_datasets(name_of_dataset_1="INTP", name_of_dataset_2="ESFJ",
                           create_language_distribution=True, create_bag_of_words=True)
