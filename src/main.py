import pandas as pd
from sklearn.linear_model import LinearRegression
from utils.load_sk_dataset import load_sk_dataset
from utils.build_chart import build_chart

wine = load_sk_dataset('wine')
wine_df = pd.DataFrame(data=wine.data, columns=wine.feature_names)

X = wine_df[['alcohol']].values
y = wine_df['malic_acid'].values

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

build_chart(X, y, y_pred, wine.target, label='Data', x_label='Alcohol', y_label='Malic Acid', title='Wine Dataset')
