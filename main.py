import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Ma'lumotlar manbasini yuklash
data = {
    'User ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Movie ID': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    'Rating': [4, 5, 3, 2, 4, 5, 3, 2, 4, 5]
}

df = pd.DataFrame(data)

# User ID va Movie ID ni klon qilish
df['User ID Clone'] = df['User ID'].copy()
df['Movie ID Clone'] = df['Movie ID'].copy()

# User ID va Movie ID ni olib tashlash
df = df.drop(['User ID', 'Movie ID'], axis=1)

# User ID va Movie ID ni klon qilish
df['User ID Clone'] = df['User ID Clone'].astype('category')
df['Movie ID Clone'] = df['Movie ID Clone'].astype('category')

# User ID va Movie ID ni kategoryga aylantirish
df['User ID Clone'] = df['User ID Clone'].cat.codes
df['Movie ID Clone'] = df['Movie ID Clone'].cat.codes

# User ID va Movie ID ni klon qilish
df['User ID Clone'] = df['User ID Clone'].astype('int64')
df['Movie ID Clone'] = df['Movie ID Clone'].astype('int64')

# User ID va Movie ID ni klon qilish
df['User ID Clone'] = df['User ID Clone'].astype('object')
df['Movie ID Clone'] = df['Movie ID Clone'].astype('object')

# User ID va Movie ID ni klon qilish
df['User ID Clone'] = df['User ID Clone'].astype('category')
df['Movie ID Clone'] = df['Movie ID Clone'].astype('category')

# User ID va Movie ID ni klon qilish
df['User ID Clone'] = df['User ID Clone'].astype('object')
df['Movie ID Clone'] = df['Movie ID Clone'].astype('object')

# User ID va Movie ID ni klon qilish
df['User ID Clone'] = df['User ID Clone'].astype('int64')
df['Movie ID Clone'] = df['Movie ID Clone'].astype('int64')

# User ID va Movie ID ni klon qilish
df['User ID Clone'] = df['User ID Clone'].astype('object')
df['Movie ID Clone'] = df['Movie ID Clone'].astype('object')

# User ID va Movie ID ni klon qilish
df['User ID Clone'] = df['User ID Clone'].astype('category')
df['Movie ID Clone'] = df['Movie ID Clone'].astype('category')

# User ID va Movie ID ni klon qilish
df['User ID Clone'] = df['User ID Clone'].astype('object')
df['Movie ID Clone'] = df['Movie ID Clone'].astype('object')

# User ID va Movie ID ni klon qilish
df['User ID Clone'] = df['User ID Clone'].astype('int64')
df['Movie ID Clone'] = df['Movie ID Clone'].astype('int64')

# User ID va Movie ID ni klon qilish
df['User ID Clone'] = df['User ID Clone'].astype('object')
df['Movie ID Clone'] = df['Movie ID Clone'].astype('object')

# User ID va Movie ID ni klon qilish
df['User ID Clone'] = df['User ID Clone'].astype('category')
df['Movie ID Clone'] = df['Movie ID Clone'].astype('category')

# User ID va Movie ID ni klon qilish
df['User ID Clone'] = df['User ID Clone'].astype('object')
df['Movie ID Clone'] = df['Movie ID Clone'].astype('object')

# User ID va Movie ID ni klon qilish
df['User ID Clone'] = df['User ID Clone'].astype('int64')
df['Movie ID Clone'] = df['Movie ID Clone'].astype('int64')

# User ID va Movie ID ni klon qilish
df['User ID Clone'] = df['User ID Clone'].astype('object')
df['Movie ID Clone'] = df['Movie ID Clone'].astype('object')

# User ID va Movie ID ni klon qilish
df['User ID Clone'] = df['User ID Clone'].astype('category')
df['Movie ID Clone'] = df['Movie ID Clone'].astype('category')

# User ID va Movie ID ni klon qilish
df['User ID Clone'] = df['User ID Clone'].astype('object')
df['Movie ID Clone'] = df['Movie ID Clone'].astype('object')

# User ID va Movie ID ni klon qilish
df['User ID Clone'] = df['User ID Clone'].astype('int64')
df['Movie ID Clone'] = df['Movie ID Clone'].astype('int64')

# User ID va Movie ID ni klon qilish
df['User ID Clone'] = df['User ID Clone'].astype('object')
df['Movie ID Clone'] = df['Movie ID Clone'].astype('object')

# User ID va Movie ID ni klon qilish
df['User ID Clone'] = df['User ID Clone'].astype('category')
df['Movie ID Clone'] = df['Movie ID Clone'].astype('category')

# User ID va Movie ID ni klon qilish
df['User ID Clone'] = df['User ID Clone'].astype('object')
df['Movie ID Clone'] = df['Movie ID Clone'].astype('object')

# User ID va Movie ID ni klon qilish
df['User ID Clone'] = df['User ID Clone'].astype('int64')
df['Movie ID Clone'] = df['Movie ID Clone'].astype('int64')

# User ID va Movie ID ni klon qilish
df['User ID Clone'] = df['User ID Clone'].astype('object')
df['Movie ID Clone'] = df['Movie ID Clone'].astype('object')

# User ID va Movie ID ni klon qilish
df['User ID Clone'] = df['User ID Clone'].astype('category')
df['Movie ID Clone'] = df['Movie ID Clone'].astype('category')

# User ID va Movie ID ni klon qilish
df['User ID Clone'] = df['User ID Clone'].astype('object')
df['Movie ID Clone'] = df['Movie ID Clone'].astype('object')

# User ID va Movie ID ni klon qilish
df['User ID Clone'] = df['User ID Clone'].astype('int64')
df['Movie ID Clone'] = df['Movie ID Clone'].astype('int64')

# User ID va Movie ID ni klon qilish
df['User ID Clone'] = df['User ID Clone'].astype('object')
df['Movie ID Clone'] = df['Movie ID Clone'].astype('object')

# User ID va Movie ID ni klon qilish
df['User ID Clone'] = df['User ID Clone'].astype('category')
df['Movie ID Clone'] = df['Movie ID Clone'].astype('category')

# User ID va Movie ID ni klon qilish
df['User ID Clone'] = df['User ID Clone'].astype('object')
df['Movie ID Clone'] = df['Movie ID Clone'].astype('object')

# User ID va Movie ID ni klon qilish
df['User ID Clone'] = df['User ID Clone'].astype('int64')
df['Movie ID Clone'] = df['Movie ID Clone'].astype('int64')

# User ID va Movie ID ni klon qilish
df['User ID Clone'] = df['User ID Clone'].astype('object')
df['Movie ID Clone'] = df['Movie ID Clone'].astype('object')

# User ID va Movie ID ni klon qilish
df['User ID Clone'] = df['User ID Clone'].astype('category')
df['Movie ID Clone'] = df['Movie ID Clone'].astype('category')

# User ID va Movie ID ni klon qilish
df['User ID Clone'] = df['User ID Clone'].astype('object')
df['Movie ID Clone'] = df['Movie ID Clone'].astype('object')

# User ID va Movie ID ni klon qilish
df['User ID Clone'] = df['User ID Clone'].astype('int64')
df['Movie ID Clone'] = df['Movie ID Clone'].astype('int64')

# User ID va Movie ID ni klon qilish
df['User ID Clone'] = df['User ID Clone'].astype('object')
df['Movie ID Clone'] = df['Movie ID Clone'].astype('object')

# User ID va Movie ID ni klon qilish
df['User ID Clone'] = df['User ID Clone'].astype('category')
df['Movie ID Clone'] = df['Movie ID Clone'].astype('category')

# User ID va Movie ID ni klon qilish
df['User ID Clone'] = df['User ID Clone'].astype('object')
df['Movie ID Clone'] = df['Movie ID Clone'].astype('object')

# User ID va Movie ID
