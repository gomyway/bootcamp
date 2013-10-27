Chapter 6 - Classification II - Sentiment Analysis
==================================================

When doing last code sanity checks for the book, Twitter
was using the API 1.0, which did not require authentication.
With its switch to version 1.1, this has now changed.
1.
If you don't have already created your personal Twitter
access keys and tokens, you might want to do so at https://dev.twitter.com/apps/new and log in

Note that some tweets might be missing when you are running install.py. 
We experimented a bit with with the tweet-fetch-rate and found that
max_tweets_per_hr=10000 works just fine, now that we are using OAuth. If you experience issues you might want to lower this value.

2.
download sentiwrodnet

http://sentiwordnet.isti.cnr.it/download.php

unzip and move to ./data/

3. download nltk data:  maxent_treebank_pos_tagger
import nltk
nltk.download('maxent_treebank_pos_tagger')
[nltk_data] Downloading package 'maxent_treebank_pos_tagger' to
[nltk_data]     C:\Users\tlj\AppData\Roaming\nltk_data...
[nltk_data]   Unzipping taggers\maxent_treebank_pos_tagger.zip.
Out[1]: True