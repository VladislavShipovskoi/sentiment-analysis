from core import load_dict,clean_tweets
from subprocess import Popen
import settings
import dataset
import tweepy
import joblib
import json


class StreamListener(tweepy.StreamListener):

    try:
        clf = joblib.load('classifier.pkl')
        vec = joblib.load('vectorizer.pkl')
    except OSError as e:
        print('Обученный классификатор не найден')

    def on_status(self, status):
        if status.retweeted:
            return

        name = status.user.screen_name
        text = status.text
        description = status.user.description
        loc = status.user.location
        coords = status.coordinates
        geo = status.geo
        user_created = status.user.created_at
        followers = status.user.followers_count
        id_str = status.id_str
        created = status.created_at
        retweets = status.retweet_count

        if geo is not None:
            geo = json.dumps(geo)

        if coords is not None:
            coords = json.dumps(coords)

        text_clean = list()
        pos_words, neg_words, stop_words, obscene, pos_emoji, neg_emoji = load_dict()
        text_clean.append(clean_tweets(text, pos_words, neg_words, obscene, pos_emoji, neg_emoji))
        X = self.vec.transform(text_clean)
        sentiment = (self.clf.predict(X[0]))[0]
        db = dataset.connect(settings.CONNECTION_STRING)
        table = db[settings.TABLE_NAME]
        try:
            table.insert(dict(
                user_name=name,
                text=text,
                sentiment=sentiment,
                user_description=description,
                user_location=loc,
                coordinates=coords,
                geo=geo,
                user_created=user_created,
                user_followers=followers,
                id_str=id_str,
                created=created,
                retweet_count=retweets,
            ))
        except Exception:
            pass

    def on_error(self, status_code):
        if status_code == 420:
            return False


if __name__ == '__main__':
    db = dataset.connect(settings.CONNECTION_STRING)
    print("Database created")
    Popen(["sqlite_web", "-p 8000", "database_tweets_stream.db"])
    auth = tweepy.OAuthHandler(settings.TWITTER_APP_KEY, settings.TWITTER_APP_SECRET)
    auth.set_access_token(settings.TWITTER_KEY, settings.TWITTER_SECRET)
    api = tweepy.API(auth)
    stream_listener = StreamListener()
    stream = tweepy.Stream(auth=api.auth, listener=stream_listener)
    stream.filter(track=settings.TRACK_TERMS, languages=['ru'])