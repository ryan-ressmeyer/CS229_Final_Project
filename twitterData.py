import json
import tweepy
import requests
import csv
import numpy as np
import pandas as pd

def createCSVFile(handle, name):
	with open('twitterKeys.json', 'r') as f:
		a = json.load(f)

	auth = tweepy.OAuthHandler(a['consumer'], a['secret'])
	auth.set_access_token("1606218469-IzazOaPsdgPu3yqkHUrElWTGJ19p27ARZnRyTMk", "bWotiZL5wV7hv4bcE6gRWIqLpO0Bt49qlMtqrklNQ74Fz")
	api = tweepy.API(auth)


	alltweets = []

		#make initial request for most recent tweets (200 is the maximum allowed count)
	new_tweets = api.user_timeline(screen_name = handle,count=200, tweet_mode = "extended")

		#save most recent tweets
	alltweets.extend(new_tweets)

		#save the id of the oldest tweet less one
	oldest = alltweets[-1].id - 1

		#keep grabbing tweets until there are no tweets left to grab
	while len(new_tweets) > 0 and len(alltweets) < 10000:
		print( "getting tweets before %s" % (oldest))

			#all subsiquent requests use the max_id param to prevent duplicates
		new_tweets = api.user_timeline(screen_name = handle, count=200,max_id=oldest, tweet_mode = "extended")

			#save most recent tweets
		alltweets.extend(new_tweets)

			#update the id of the oldest tweet less one
		oldest = alltweets[-1].id - 1

		print("...%s tweets downloaded so far" % (len(alltweets)))

		#transform the tweepy tweets into a 2D array that will populate the csv

	def func(var):
		return not var[2][:2] == 'RT'
	outtweets = [[tweet.id_str, tweet.created_at, tweet.full_text] for tweet in alltweets]
	ot = filter(func, list(outtweets))


	rez = np.array(list(ot))[:,2]

	last = [''.join(w.split('\n')) for w in rez]
	lastLast = [" ".join(w.split()[:-1]) if 'https' in w.split()[-1] else w for w in last]
	# print(lastLast[::-1])

	a = np.repeat(name, len(lastLast))
	df = pd.DataFrame(np.array([a, lastLast]).T, columns=["Name", "Tweet"])
	pd.set_option("display.max_colwidth", 10000)
	df.to_csv(name + ".csv")


createCSVFile("SenWarren", "Elizabeth Warren")
# createCSVFile("SenWarren", "Elizabeth Warren")
# createCSVFile("SenWarren", "Elizabeth Warren")
# createCSVFile("SenWarren", "Elizabeth Warren")
# createCSVFile("SenWarren", "Elizabeth Warren")

