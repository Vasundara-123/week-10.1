
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# Load the coffee dataset
url = url = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"
df = pd.read_csv(url)

# Exercise 1: Linear Regression
data1 = df[["100g_USD", "rating"]].dropna()

X1 = data1[["100g_USD"]]
y1 = data1["rating"]

lr = LinearRegression()
lr.fit(X1, y1)

with open("model_1.pickle", "wb") as f:
    pickle.dump(lr, f)

print("Exercise 1 completed")


# Exercise 2: Convert roast into numeric categories
def roast_category(roast):
    categories = {
        "Light": 0,
        "Medium-Light": 1,
        "Medium": 2,
        "Medium-Dark": 3,
        "Dark": 4
    }
    return categories.get(roast, float("nan"))


df["roast_cat"] = df["roast"].apply(roast_category)

X2 = df[["100g_USD", "roast_cat"]]
y2 = df["rating"]

# Remove missing targets; tree can handle missing features
valid = y2.notna()
X2 = X2.loc[valid]
y2 = y2.loc[valid]

dtr = DecisionTreeRegressor(random_state=42)
dtr.fit(X2, y2)

with open("model_2.pickle", "wb") as f:
    pickle.dump(dtr, f)

print("Exercise 2 completed")
