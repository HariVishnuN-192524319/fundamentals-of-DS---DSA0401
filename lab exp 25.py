import pandas as pd
from scipy import stats
import numpy as np

df = pd.read_csv("customer_reviews.csv")

ratings = df["Rating"]

n = len(ratings)
mean = ratings.mean()
std = ratings.std()

confidence = 0.95
alpha = 1 - confidence

t_value = stats.t.ppf(1 - alpha / 2, n - 1)

margin_error = t_value * (std / np.sqrt(n))

lower = mean - margin_error
upper = mean + margin_error

print("Customer Reviews Analysis")
print("-------------------------")

print("Number of Reviews:", n)
print("Mean Rating:", round(mean, 2))
print("Standard Deviation:", round(std, 2))

print("\n95% Confidence Interval")
print("Lower Limit:", round(lower, 2))
print("Upper Limit:", round(upper, 2))
