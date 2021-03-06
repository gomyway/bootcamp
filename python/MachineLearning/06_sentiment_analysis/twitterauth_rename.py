#rename this script to twitterauth.py and put key/secrets here
import sys

CONSUMER_KEY =  None
CONSUMER_SECRET = None

ACCESS_TOKEN_KEY = None
ACCESS_TOKEN_SECRET = None

if CONSUMER_KEY is None or CONSUMER_SECRET is None or ACCESS_TOKEN_KEY is None or ACCESS_TOKEN_SECRET is None:
    print("""\
When doing last code sanity checks for the book, Twitter
was using the API 1.0, which did not require authentication.
With its switch to version 1.1, this has now changed.

It seems that you don't have already created your personal Twitter
access keys and tokens. Please do so at
Go to https://dev.twitter.com/apps/new and log in, if necessary
and paste the keys/secrets into twitterauth.py

Sorry for the inconvenience,
The authors.""")

    sys.exit(1)
