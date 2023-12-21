from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import tensorflow as tf

model=load_model('satu_saved_model')

## Input Data
print("Masukkan User ")
user_pilih = int(input())
print("Kota Tujuan ")
kota_terpilih = input()

# Import Data
rating = pd.read_csv('./dataset/tourism_rating.csv')
place = pd.read_csv('./dataset/tourism_with_id.csv')
user = pd.read_csv('./dataset/user.csv')

#Merge Data
rating = pd.merge(rating, place[['Place_Id']], how='right', on='Place_Id')
user = pd.merge(user, rating[['User_Id']], how='right', on='User_Id').drop_duplicates().sort_values('User_Id')
place = pd.read_csv('./dataset/tourism_with_id.csv')
place = place.drop(['Unnamed: 11','Unnamed: 12'],axis=1)

place_df = place[['Place_Id','Place_Name','Category','Rating','Price']]
place_df.columns = ['id','place_name','category','rating','price']


place = place[place['City'].isin([kota_terpilih])]

df=rating.copy()
user_unique_vals = df['User_Id'].unique().tolist()
user_to_user_encoded = {x: i for i, x in enumerate(user_unique_vals)}
user_encoded_to_user = {i: x for i, x in enumerate(user_unique_vals)}
df['user'] = df['User_Id'].map(user_to_user_encoded)

place_unique_vals = df['Place_Id'].unique().tolist()
place_to_place_encoded = {x: i for i, x in enumerate(place_unique_vals)}
place_encoded_to_place = {i: x for i, x in enumerate(place_unique_vals)}
df['place'] = df['Place_Id'].map(place_to_place_encoded)

# Mendapatkan jumlah user dan place
num_users, num_place = len(user_to_user_encoded), len(place_to_place_encoded)

# Mengubah rating menjadi nilai float
df['Place_Ratings'] = df['Place_Ratings'].values.astype(np.float32)

# Mendapatkan nilai minimum dan maksimum rating
min_rating, max_rating = min(df['Place_Ratings']), max(df['Place_Ratings'])

print(f'Number of User: {num_users}, Number of Place: {num_place}, Min Rating: {min_rating}, Max Rating: {max_rating}')

place_df = place[['Place_Id','Place_Name','Category','Rating','Price']]
place_df.columns = ['id','place_name','category','rating','price']
df = rating.copy()

#enconding
user_id = user_pilih
place_visited_by_user = df[df.User_Id == user_id]

# Membuat data lokasi yang belum dikunjungi user
place_not_visited = place_df[~place_df['id'].isin(place_visited_by_user.Place_Id.values)]['id']
place_not_visited = list(
    set(place_not_visited)
    .intersection(set(place_to_place_encoded.keys()))
)

place_not_visited = [[place_to_place_encoded.get(x)] for x in place_not_visited]
user_encoder = user_to_user_encoded.get(user_id)
user_place_array = np.hstack(
    ([[user_encoder]] * len(place_not_visited), place_not_visited)
)

# Memberikan 20 Rekomendasi Tempat wisata
user_place_array = tf.cast(user_place_array, tf.int64)
ratings = model.predict(user_place_array).flatten()

top_ratings_indices = ratings.argsort()[-21:][::-1]
recommended_place_ids = [
    place_encoded_to_place.get(place_not_visited[x][0]) for x in top_ratings_indices
]

print('Daftar rekomendasi untuk: {}'.format('User ' + str(user_id)))
print('===' * 15, '\n')
print('----' * 15)
print('Tempat dengan rating wisata paling tinggi dari user')
print('----' * 15)

top_place_user = (
    place_visited_by_user.sort_values(
        by='Place_Ratings',
        ascending=False
    )
    .head(10)
    .Place_Id.values
)

place_df_rows = place_df[place_df['id'].isin(top_place_user)]
for row in place_df_rows.itertuples():
    print(row.place_name, ':', row.category)

print('')
print('----' * 15)
print('Top 20 place recommendations for you in ' +  kota_terpilih )
print('----' * 15)

recommended_place = place_df[place_df['id'].isin(recommended_place_ids)]
for row, i in zip(recommended_place.itertuples(), range(1, 21)):
    print(i, '.', row.place_name, '\n    ', row.category, ',', 'Harga Tiket Masuk ', row.price, ',',
          'Rating Wisata ', row.rating, '\n')

print('===' * 15)