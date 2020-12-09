import pystan
import moderation
import pandas as pd
import numpy as np
import pickle
import math
from functools import reduce
from operator import add

def inv_logit(x):
	return(1/(1 + math.exp(-x)))

def flatten(lst):
	x = [lst] if isinstance(lst, float) or isinstance(lst, int) else reduce(add, [flatten(ele) for ele in lst])
	return(x)

# `compute_weights` function which estimates personalised weights for each category using a 
# binomial hierarchical logistic regression model
# 
# input: 
# 	- a dataframe of tweets with columns user_id, tweet_label (offensive: 1, inoffensive: 0), 
#     perspective scores for flirtation, identity_attack, sexually_explicit, threat, toxicity
#   - input df should be read from a database to create this dataframe, which can then be 
#     passed as argument to `compute_weights`
# output: 
#   - a dataframe with columns user_id, and weights w1 - w5 corresponding to 
#     flirtation, identity_attack, sexually_explicit, threat, toxicity for each user 
#   - this can be passed to a function and stored in a database
def compute_weights(df_tweet, path_to_model_file, _iter = 5000, _samples = 2000, _chains = 4, _cores = 4):
	x_preds = ['flirtation', 'identity_attack', 'sexually_explicit', 'threat', 'toxicity']

	tweet_dat = {
		'N': len(df_tweet),
		'Y': df_tweets['offensive'],
		'K': len(x_preds),
		'X': np.array(df_tweet.loc[:, x_preds]),
		'Z_1': df_tweet[x_preds[0]],
		'Z_2': df_tweet[x_preds[1]],
		'Z_3': df_tweet[x_preds[2]],
		'Z_4': df_tweet[x_preds[3]],
		'Z_5': df_tweet[x_preds[4]],
		'J': len(df_tweet.user_id.unique()),
		'worker_Id': list(df_tweet.user_id),
		'NC_1': 10
	}

	binom_model = pickle.load(model_file, open(path_to_model_file, 'rb'))

	fit2_binom = binom_model.sampling(data = new_data, iter = _iter, warmup = (_iter - _samples), chains = _chains, cores = _cores, control = {'adapt_delta': 0.98})

	draws = fit2_binom.extract(pars = ["b", "r_1", "r_2", "r_3", "r_4", "r_5"])
	fixed_effs = pd.DataFrame(draws["b"], columns=['b_1', 'b_2', 'b_3', 'b_4', 'b_5'])
	fixed_effs.insert(0, '.draw', pd.Series(range(1, (_samples*_chains + 1))).values)
	ran_effs = pd.DataFrame.from_dict(dict((k, flatten(draws[k])) for k in ("r_1", "r_2", "r_3", "r_4", "r_5")))
	ran_effs.insert(0, '.draw', pd.concat([pd.Series(range(1, (_samples*_chains + 1)))] * new_data['J']).values)
	ran_effs.insert(1, 'user_id', pd.Series(range(1,(new_data['J'] + 1))).repeat((_samples*_chains)).values)

	fitted_draws = fixed_effs.merge(ran_effs, on = ".draw", how = "right")

	fitted_draws = fitted_draws.assign(
		w1 = (fitted_draws['b_1']+fitted_draws['r_1']),
		w2 = (fitted_draws['b_2']+fitted_draws['r_2']),
		w3 = (fitted_draws['b_3']+fitted_draws['r_3']),
		w4 = (fitted_draws['b_4']+fitted_draws['r_4']),
		w5 = (fitted_draws['b_5']+fitted_draws['r_5']),
	)

	weight_df = fitted_draws[['.draw', 'user_id', 'w1', 'w2', 'w3', 'w4', 'w5']].groupby(
			['user_id']
		).agg(
			{'w1': 'median', 'w2': 'median', 'w3': 'median', 'w4': 'median', 'w5': 'median'}
		)

	return(weight_df)

# `predict` function which estimates probability of a tweet being offensive based on
# personalised weights for each user
# 
# input: 
# 	- a dictionary of weights for a particular user
#   - a dictionary of probability scores in each category from perspective API 
#     for a particular tweet
# output: 
#   - a predicted probability value (in [0, 1]) of the tweet being deemed offensive by the user
def predict(tweet_scores, user_weight):
	p = (user_weight['w1']*tweet_scores['flirtation'] + 
		user_weight['w2']*tweet_scores['identity_attack'] + 
		user_weight['w3']*tweet_scores['sexually_explicit'] + 
		user_weight['w4']*tweet_scores['threat'] + 
		user_weight['w5']*tweet_scores['toxicity']
	)

	_value = inv_logit(p)

	return(_value)




