import os, sys
from tqdm import tqdm

# For data Handling
import pandas as pd
import numpy as np


# Ignore Warnings
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans

args = sys.argv

if len(args) < 3:
    print("USAGE: <songname> <top_n>")
    exit(1)


# load and preprocess dataset
def load_data():
    df = pd.read_csv('data.csv')
    df["artists"] = df["artists"].str.replace("[", "").str.replace("]", "").str.replace("'", "")
    
    num_types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    num = df.select_dtypes(include=num_types)
    num = num.fillna(num.mean())
    
    # normalize numeric columns
    def normalize_column(col):
        max_d = df[col].max()
        min_d = df[col].min()
        df[col] = (df[col] - min_d) / (max_d - min_d)
    
    for col in num.columns:
        normalize_column(col)
    
    km = KMeans(n_clusters=25)
    pred = km.fit_predict(num)
    df['pred'] = pred
    normalize_column('pred')
    
    return df

class SongRecommender:
    def __init__(self, data):
        self.data_ = data
    
    def get_recommendations(self, song_name, n_top):
        '''
        Get song recommendations based on the similarity to a given song.
        '''
        distances = []
        
        song = self.data_[(self.data_.name.str.lower() == song_name.lower())].head(1).values[0]
        rem_data = self.data_[self.data_.name.str.lower() != song_name.lower()]

        for r_song in tqdm(rem_data.values):
            dist = 0
            for col in np.arange(len(rem_data.columns)):
                if col not in [3, 8, 14, 16]:
                    dist += np.absolute(float(song[col]) - float(r_song[col]))
            distances.append(dist)
        
        rem_data['distance'] = distances
        rem_data = rem_data.sort_values('distance')

        columns = ['artists', 'name']
        top_recommendations = rem_data[columns].head(n_top)
        
        result = []
        for i, row in top_recommendations.iterrows():
            result.append(f"Artist: {row['artists']}, Song: {row['name']}")
        
        return '\n'.join(result)


def main():
    df = load_data()
    recommender = SongRecommender(df)
    
    song_name = args[1]
    n_top = int(args[2])
    
    recommendations = recommender.get_recommendations(song_name, n_top)
    print(f"Recommendations for '{song_name}':\n{recommendations}")


if __name__ == "__main__":
    main()
    # pass
