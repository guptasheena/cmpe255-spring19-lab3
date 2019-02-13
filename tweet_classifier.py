def load_data():
    data = []
    data_labels = []
    with open("./pos_tweets.txt") as f:
        for i in f:
            data.append(i)
            data_labels.append('pos')

    with open("./neg_tweets.txt") as f:
        for i in f:
            data.append(i)
            data_labels.append('neg')

    return data, data_labels


def transform_to_features(data):
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(
        analyzer='word',
        lowercase=False,
    )
    features = vectorizer.fit_transform(
        data
    )
    features_nd = features.toarray()
    return features_nd


def train_then_build_model(data_labels, features_nd, data):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        features_nd,
        data_labels,
        train_size=0.80,
        random_state=1234)

    X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
        data,
        data_labels,
        train_size=0.80,
        random_state=1234)

    from sklearn.linear_model import LogisticRegression
    log_model = LogisticRegression()

    log_model = log_model.fit(X=X_train, y=y_train)
    y_pred = log_model.predict(X_test)

    # print first 10th prediction in this format:
    # ::{prediction}::{tweet}
    for value in range(0, 10):
        print("::"+str(y_pred[value])+"::"+str(X_test_new[value]))

    # print accuracy
    from sklearn.metrics import accuracy_score
    print("Accuracy="+str(accuracy_score(y_test, y_pred)))
    # print("Accuracy={}".format('fix-me'))


def process():
    data, data_labels = load_data()
    features_nd = transform_to_features(data)
    train_then_build_model(data_labels, features_nd, data)


process()
