import json
import urllib.request

db_url = "http://127.0.0.1:5000/api/"
articles_db_url = db_url + "articles"


def get_articles():
    response = urllib.request.urlopen(articles_db_url)
    data = response.read()
    response_dict = json.loads(data)
    return response_dict
