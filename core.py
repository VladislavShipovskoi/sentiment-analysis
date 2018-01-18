# -*- coding: utf-8 -*-
import csv
import random
import re
import string
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymorphy2
from nltk.tokenize import TweetTokenizer
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def clean_tweets(a, pos_words, neg_words, obscene, pos_emoji, neg_emoji):
    tknzr = TweetTokenizer()
    a = tknzr.tokenize(a)

    for n, i in enumerate(a):
        if i in pos_emoji:
            a[n] = ' положительныйэмотикон '
        if i in neg_emoji:
            a[n] = ' негативныйэмотикон '
        if i in pos_words:
            a[n] = ' положительноеслово '
        if i in neg_words:
            a[n] = ' негативноеслово '
        if i in obscene:
            a[n] = ' обсценнаялексика '

    a = ' '.join(a)
    result = re.sub(r'(?:@[\w_]+)', '', a)  # упоминания
    result = re.sub(r'(?:\#+[\w_]+[\w\'_\-]*[\w_]+)', '', result)  # хештеги
    result = re.sub(r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', '', result)  # ссылки
    result = re.sub(r'RT', '', result)  # RT
    result = re.sub(r'(?:(?:\d+,?)+(?:\.?\d+)?)', ' ', result)  # цифры
    result = re.sub(r'[^а-яеёА-ЯЕЁ0-9-_*.]', ' ', result)  # символы
    result = re.sub(r'[a-zA-Z.,?!@#$%^&*()_+-]+', ' ', result)  # англ слова и символы
    # result = ''.join(ch for ch, _ in itertools.groupby(result))  # повторяющиеся буквы
    result = result.lower()  # приведение к низкому регистру
    result = re.sub(r'\s+', ' ', result)  # лишние пробелы
    cleantweet = ' '.join(word for word in result.split() if len(word) > 2)  # удаление слов длинной 1,2 символа
    cleantweet = cleantweet.strip()

    return cleantweet


def load_dict():
    with open('data/dictionary/pos_words.txt') as f:
        pos_words = f.read().splitlines()

    with open('data/dictionary/neg_words.txt') as f:
        neg_words = f.read().splitlines()

    with open('data/dictionary/stop_words.txt') as f:
        stop_words = f.read().splitlines()

    with open('data/dictionary/obscene.txt') as f:
        obscene_words = f.read().splitlines()

    with open('data/dictionary/possmile.txt') as f:
        pos_emoji = f.read().splitlines()

    with open('data/dictionary/negsmile.txt') as f:
        neg_emoji = f.read().splitlines()

    return pos_words, neg_words, stop_words, obscene_words, pos_emoji, neg_emoji


def load_data():
    data, text, sentiment = list(), list(), list()
    pos_words, neg_words, stop_words, obscene, pos_emoji, neg_emoji = load_dict()

    neu_csv_file = open('data/tweets/neutral.csv', "r", encoding='utf-8')
    reader = csv.reader(neu_csv_file)
    for row in reader:
        row = ' '.join(row)
        cleanrow = clean_tweets(row, pos_words, neg_words, obscene, pos_emoji, neg_emoji)
        data.append([cleanrow, '0'])

    pos_csv_file = open('data/tweets/positive.csv', "rt", encoding='utf-8')
    reader = csv.reader(pos_csv_file)
    for row in reader:
        row = ' '.join(row)
        cleanrow = clean_tweets(row, pos_words, neg_words, obscene, pos_emoji, neg_emoji)
        data.append([cleanrow, '1'])

    neg_csv_file = open('data/tweets/negative.csv', "rt", encoding='utf-8')
    reader = csv.reader(neg_csv_file)
    for row in reader:
        row = ' '.join(row)
        cleanrow = clean_tweets(row, pos_words, neg_words, obscene, pos_emoji, neg_emoji)
        data.append([cleanrow, '-1'])

    random.shuffle(data)
    for i in data:
        text.append(i[0])
        sentiment.append(i[1])

    return text, sentiment


def cross_validation(x, y, vector, classifier):
    x_folds = np.array_split(x, 10)
    y_folds = np.array_split(y, 10)
    score_train = list()
    score_test = list()
    for k in range(5):
        print(k)
        x_train = list(x_folds)
        x_test = x_train.pop(k)
        x_train = np.concatenate(x_train)
        y_train = list(y_folds)
        y_test = y_train.pop(k)
        y_train = np.concatenate(y_train)
        x_train = vector.fit_transform(x_train)
        x_test = vector.transform(x_test)
        score_train.append(classifier.fit(x_train, y_train).score(x_train, y_train))
        score_test.append(classifier.fit(x_train, y_train).score(x_test, y_test))
        y_predicted = classifier.predict(x_test)
        print("*****")
        print("Отчет классификации - %s" % classifier)
        print(metrics.classification_report(y_test, y_predicted))
    return score_train, score_test


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1, 10)):

    font = {'family': 'Droid Sans',
            'weight': 'normal',
            'size': 14}
    plt.rc('font', **font)
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel(u"Объем тренировочной выборки")
    plt.ylabel(u"Точность")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                                                            train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label=u"Точность обучения")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label=u"Точность тестирования")

    plt.legend(loc="best")
    print(pd.DataFrame(train_scores))
    print(pd.DataFrame(test_scores))
    return plt


def learning_curves(title, x, y, estimator):
    cv = ShuffleSplit(n_splits=10, test_size=0.4)
    plot_learning_curve(estimator, title, x, y, (0.4, 1.02), cv=cv, n_jobs=-1)
    plt.show()


def term_freq(x, vector):
    word_freq_df = pd.DataFrame(
        {'term': vector.get_feature_names(), 'occurrences': np.asarray(x.sum(axis=0)).ravel().tolist()})
    word_freq_df['frequency'] = word_freq_df['occurrences'] / np.sum(word_freq_df['occurrences'])
    word_freq_df.sort_values('occurrences', ascending=False).to_csv('data/term_frequences.csv')


def save_model(x, y, vector, clf):
    vec_clf = Pipeline([('vectorizer', vector), ('classifier', clf)])
    vec_clf.fit(x, y)
    joblib.dump(clf, filename="classifier.pkl")
    joblib.dump(vector, filename="vectorizer.pkl")


def sentiment_analysis(text, sentiment):
    token_pattern = r'\w+|[%s]' % string.punctuation
    vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=token_pattern, binary=False, min_df=5)
    X = vectorizer.fit_transform(text)
    print("Объем словаря: %s" % len(vectorizer.vocabulary_))
    classifier = LinearSVC()
    print("Обучение модели")
    term_freq(X,vectorizer)
    learning_curves(u'Кривая обучения', X, sentiment, classifier)
    X_train, X_test, y_train, y_test = train_test_split(X, sentiment, test_size=0.4)
    save_model(text, sentiment, vectorizer, classifier)
    classifier.fit(X_train, y_train)
    prediction = classifier.predict(X_test)
    print('accuracy_score ' + str(100*(accuracy_score(y_test, prediction))) + '%')
    print(metrics.classification_report(y_test, prediction))


if __name__ == '__main__':
    pos_words, neg_words, stop_words, obscene, pos_emoji, neg_emoji = load_dict()
    text, sentiment = load_data()
    sentiment_analysis(text, sentiment)