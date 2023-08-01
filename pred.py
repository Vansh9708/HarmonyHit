pip install spotipy
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

#Authentication - without user
client_credentials_manager = SpotifyClientCredentials(client_id = '3359221a90cd4f3f842b537fd1917879', client_secret = '1a5d31cc93d24d45acfb13c949255eea')
sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)

playlist_link ="https://open.spotify.com/playlist/5JlDLT9hhkanuYNigN8bsO?si=44f27e56fc95479c"

playlist_URI = playlist_link.split("/")[-1].split("?")[0]
track_uris = [x["track"]["uri"] for x in sp.playlist_tracks(playlist_URI)["items"]]
track_uris[:10]

sp.playlist_tracks(playlist_URI).get('items')[0].keys()

type(sp.playlist_tracks(playlist_URI).get('items'))
sp.playlist_tracks(playlist_URI).get('items')[0].keys()

obj = sp.playlist_tracks(playlist_URI).get('items')[0]
print(obj.get('track'))

for track in sp.playlist_tracks(playlist_URI)["items"]:
    #URI
    track_uri = track["track"]["uri"]
    
    #Track name
    track_name = track["track"]["name"]
    
    #Main Artist
    artist_uri = track["track"]["artists"][0]["uri"]
    artist_info = sp.artist(artist_uri)
    
    #Name,popularity , genre
    artist_name = track["track"]["artists"][0]["name"]
    artist_pop = artist_info["popularity"]
    artist_genres = artist_info["genres"]
    
    #Album
    album = track["track"]["album"]["name"]
    
    #Popularity of the track
    track_pop = track["track"]["popularity"]
    
    #printing
    print(track_name,artist_uri,artist_name,artist_pop,album,track_pop)
  # Extracting features from the tracks
sp.audio_features(track_uris)[0]

# Creating dataframe to store these values

import pandas as pd
track_feature = pd.DataFrame(sp.audio_features(track_uris))
track_feature.head()

track_uri = []
track_name = []
artist_info = []
artist_name = []
artist_pop = []
artist_genres = []
track_pop = []
album = []


for track in sp.playlist_tracks(playlist_URI)["items"]:
    #URI
    track_uri.append(track["track"]["uri"])
    
    #Track name
    track_name.append(track["track"]["name"])
    
    #Main Artist
    artist_uri = track["track"]["artists"][0]["uri"]
    ar_info = sp.artist(artist_uri)
    artist_info.append(ar_info)
    
    #Name, popularity , genre
    artist_name.append(track["track"]["artists"][0]["name"])
    artist_pop.append(ar_info["popularity"])
    artist_genres.append(ar_info["genres"])
    
    #Album
    album.append(track["track"]["album"]["name"])
    
    #Popularity of the track
    track_pop.append(track["track"]["popularity"])
  extra_data = pd.DataFrame(list(zip(track_uri,track_name,artist_info,artist_name,artist_pop,artist_genres,track_pop,album)), columns =['Track_identifier','Track_name','Artist information', 'Artist_name','Artist Pop', 'Genre of Artist','Track Pop','Album_name'])
extra_data.head()

#Merging the dataframe to get the whole data
finalData = pd.concat([extra_data,track_feature],axis = 1)
finalData.head()

# Saving my data:
# finalData.to_csv
finalData.to_csv('spotify.csv',index =False)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier

df = pd.read_csv('spotify.csv')
df.head()
df.shape
df.isnull().sum()
drop_list = ['Artist_name','Artist information','Track_identifier','Track_name','Album_name','key','mode','Genre of Artist',]
train = df.drop(drop_list, axis=1)
train1 = train.drop(['id'], axis = 1)
train2 = train1.drop(['analysis_url', 'track_href', 'uri', 'type', ], axis = 1)
train2.head()
Y = train2['Track Pop']
X_train, X_test, y_train, y_test = train_test_split(train2, Y, test_size=0.3, random_state=0)

clf1 = MLPClassifier(hidden_layer_sizes=(200,150,50), max_iter=200,activation = 'relu',solver='adam',random_state=1)
clf1.fit(X_train, y_train)

print(clf1.score(X_train, y_train))

y_pred = clf1.predict(X_test)
clf1.score(X_test, y_test)
print(classification_report(y_test, y_pred))
