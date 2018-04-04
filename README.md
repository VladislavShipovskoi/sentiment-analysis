### Sentiment analysis russians tweets
This project determines the sentiment of Russian-language tweets.
* data/ - tweets and dictionaries
* core.py - core functional
* requirements.txt - necessary libraries
* settings.py - settings file for live stream
* tweet_stream_listener.py - live stream russian tweets and sentiment analysis
### startup instruction
1. virtualenv -p python3 venv
2. pip install -r requirements.txt
3. sudo apt-get install python3-tk
4. python core.py
### startup instruction live stream
1. Create a twitter account if you do not already have one.
2. Go to [Twitter Dev](https://apps.twitter.com/) and log in with your twitter credentials.
3. Click "Create New App"
4. Fill out the form and agree to the terms. Put in a dummy website if you don't have one you want to use.
5. On the next page, click the "API Keys" tab along the top, then scroll all the way down until you see the section "Your Access Token"
6. Click the button "Create My Access Token". You can Read more about Oauth authorization.
7. You will now copy four values into **settings.py**. These values are your "API Key", your "API secret", your "Access token" and your "Access token secret". All four should now be visible on the API Keys page. (You may see "API Key" referred to as "Consumer key" in some places in the code or on the web; they are synonyms.)
8. Open **settings.py** and set the variables corresponding to the api key, api secret, access token, and access secret. You will see code like the below:
* TWITTER_APP_KEY = "Enter api key"
* TWITTER_APP_SECRET = "Enter api secret"
* TWITTER_KEY = "Enter your access token key here"
* TWITTER_SECRET = "Enter your access token secret here"
9. python tweet_stream_listener.py

### Results
![1](https://user-images.githubusercontent.com/17500704/38318859-98ee4c96-385a-11e8-9396-1ef00ca71c15.png)
![2](https://user-images.githubusercontent.com/17500704/38318155-ecc59e5c-3858-11e8-85eb-26ad672e97b6.png)
