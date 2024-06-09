import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA

data = pd.read_csv("kc_house_data.csv")

print(data.head())
print(data.describe())

data['bedrooms'].value_counts().plot(kind='bar')
plt.title('Number of Bedrooms')
plt.xlabel('Bedrooms')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10, 10))
sns.jointplot(x=data.lat.values, y=data.long.values, size=10)
plt.ylabel('Longitude', fontsize=12)
plt.xlabel('Latitude', fontsize=12)
plt.show()

plt.scatter(data.price, data.sqft_living)
plt.title("Price vs Square Feet")
plt.show()

plt.scatter(data.price, data.long)
plt.title("Price vs Location of the area")
plt.show()

plt.scatter(data.price, data.lat)
plt.xlabel("Price")
plt.ylabel('Latitude')
plt.title("Latitude vs Price")
plt.show()

plt.scatter(data.bedrooms, data.price)
plt.title("Bedroom and Price ")
plt.xlabel("Bedrooms")
plt.ylabel("Price")
plt.show()

plt.scatter((data['sqft_living']+data['sqft_basement']), data['price'])
plt.show()

plt.scatter(data.waterfront, data.price)
plt.title("Waterfront vs Price (0= no waterfront)")
plt.show()

plt.scatter(data.floors, data.price)
plt.show()

plt.scatter(data.condition, data.price)
plt.show()

plt.scatter(data.zipcode, data.price)
plt.title("Which is the pricey location by zipcode?")
plt.show()

reg = LinearRegression()
clf = GradientBoostingRegressor(n_estimators=400, max_depth=5, min_samples_split=2,
                                learning_rate=0.1, loss='squared_error')

labels = data['price']
conv_dates = [1 if values == 2014 else 0 for values in data.date]
data['date'] = conv_dates
train1 = data.drop(['id', 'price'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(train1, labels, test_size=0.10, random_state=2)

reg.fit(x_train, y_train)
print("Linear Regression Score:", reg.score(x_test, y_test))

clf.fit(x_train, y_train)
print("Gradient Boosting Regressor Score:", clf.score(x_test, y_test))

pca = PCA()
pca_data = pca.fit_transform(scale(train1))
print(pca_data)
