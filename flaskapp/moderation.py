import json
import requests
from urllib.error import HTTPError
import sys

api_key = 'AIzaSyD6lXIN2SuT-2jMeXFvLA4-Hevc9viKgts'
url = ('https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze' +'?key=' + api_key)

## a funcition that takes a tweet (string) and output a dictionary of all the scores from perspective API
def get_perspective_score(tweet):
  score_dict = {}
  data_dict = {
    'comment': {'text': tweet},
    'languages': ['en'],
    'requestedAttributes': {'TOXICITY': {}, 'IDENTITY_ATTACK':{}, 'INSULT':{}, 'PROFANITY':{}, 'THREAT':{}, 'SEXUALLY_EXPLICIT':{},'FLIRTATION':{}}
  }
  try:
    response = requests.post(url=url, data=json.dumps(data_dict))
    response.raise_for_status()
    response_dict = json.loads(response.content)
    for key in response_dict['attributeScores']:
      value = response_dict['attributeScores'][key]['summaryScore']['value']
      key = key.lower()
      score_dict[key] = value
  except HTTPError as http_err:
    print(f'HTTP error occurred: {http_err}')
  except Exception as err:
    print(f'Other error occurred: {err}')

  return score_dict
