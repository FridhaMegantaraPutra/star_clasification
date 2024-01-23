# -*- coding: utf-8 -*-


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets, linear_model, metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

stars = pd.read_csv('Stars.csv')

stars.head(10)

index = stars.index
index.name = 'Index'

stars.head(3)

stars.columns

stars.info()

stars.isna().sum()

stars.describe()

stars.skew()

stars.duplicated().sum()

sns.heatmap(stars.corr(),
            cmap='Blues')

stars.columns

for i in ['Color', 'Spectral_Class', 'Type']:
    print("\n", i, " --->\n", stars[i].unique())
    print("\n", i, " --->\n", stars[i].value_counts())

stars['Color'] = stars['Color'].replace('Orange-Red', 'Red')
stars['Color'] = stars['Color'].replace('Pale yellow orange', 'Orange')
stars['Color'] = stars['Color'].replace('Blue White', 'Blue-White')
stars['Color'] = stars['Color'].replace('Blue white', 'Blue-White')
stars['Color'] = stars['Color'].replace('Blue-white', 'Blue-White')
stars['Color'] = stars['Color'].replace('Whitish', 'White')
stars['Color'] = stars['Color'].replace('white', 'White')
stars['Color'] = stars['Color'].replace('yellow-white', 'White-Yellow')
stars['Color'] = stars['Color'].replace('Yellowish White', 'White-Yellow')
stars['Color'] = stars['Color'].replace('yellowish', 'White-Yellow')
stars['Color'] = stars['Color'].replace('Yellowish', 'White-Yellow')

for i in ['Color', 'Spectral_Class', 'Type']:
    print("\n", i, " --->\n", stars[i].unique())
    print("\n", i, " --->\n", stars[i].value_counts())

fig = px.histogram(stars, 'Temperature',
                   color='Color',
                   title="<b>Average Temparature by color</b>")

fig.add_vline(x=stars['Temperature'].mean(), line_width=2,
              line_dash="dash", line_color="black")

fig.show()

# balanced data
stars.groupby('Type').count().plot(kind='bar')

# plotting colors by frequencies
color = ['Red', 'SteelBlue', 'lightskyblue', 'lightyellow', 'White', 'Orange']
stars['Color'].value_counts().plot(
    kind='bar', color=color, edgecolor='slategray')
plt.xticks(rotation=45)
plt.title('Red colored stars are highest in number')
plt.xlabel('Color')
plt.ylabel('Frequency')

stars['Spectral_Class'].value_counts().plot(
    kind='bar', color='hotpink', edgecolor='mediumorchid')
plt.title('Most of the stars have Classs M')
plt.xlabel('Spectral Class')
plt.ylabel('Frequency')
plt.xticks(rotation=0)

# Most of the stars have luminosity close to 25k with blue stars having highest number of luminous stars
fig = px.histogram(stars, 'L',
                   color='Color',
                   title="<b>Average Luminosity by color</b>")

fig.add_vline(x=stars['L'].mean(), line_width=2,
              line_dash="dash", line_color="black")

fig.show()

# Most of the stars have small relative radius close to 100 with blue stars having highest number of larger stars
fig = px.histogram(stars, 'R',
                   color='Color',
                   title="<b>Average Relative Radius by color</b>")

fig.add_vline(x=stars['R'].mean(), line_width=2,
              line_dash="dash", line_color="black")

fig.show()

sns.barplot(data=stars, x='Type', y='Temperature')

sns.barplot(data=stars, x='Color', y='Temperature')

sns.barplot(data=stars, x='Spectral_Class', y='Temperature')

# doesn't show much
sns.pairplot(stars)

stars_type_0 = stars[stars['Type'] == 0]
stars_type_1 = stars[stars['Type'] == 1]
stars_type_2 = stars[stars['Type'] == 2]
stars_type_3 = stars[stars['Type'] == 3]
stars_type_4 = stars[stars['Type'] == 4]
stars_type_5 = stars[stars['Type'] == 5]

star_type_df = [stars_type_0, stars_type_1, stars_type_2,
                stars_type_3, stars_type_4, stars_type_5]
star_type_names_df = ['0', '1', '2', '3', '4', '5']

for index, star_type in enumerate(star_type_df):
    plt.figure(figsize=(10, 8))
    plt.hist(star_type['Color'],
             bins=10)
    plt.title('Colors for star Type : ' + star_type_names_df[index])
    plt.xlabel('Color')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)

    plt.show()

for index, star_type in enumerate(star_type_df):
    plt.figure(figsize=(10, 8))
    sns.histplot(star_type['Temperature'],
                 bins=25, kde=True)
    plt.title('Temperature for star Type : ' + star_type_names_df[index])
    plt.xlabel('Temperature')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)

    plt.show()

for index, star_type in enumerate(star_type_df):
    plt.figure(figsize=(10, 8))
    plt.hist(star_type['Spectral_Class'])
    plt.title('Spectral Classes for star Type : ' + star_type_names_df[index])
    plt.xlabel('spectral Class')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)

    plt.show()

for index, star_type in enumerate(star_type_df):
    plt.figure(figsize=(10, 8))
    sns.histplot(star_type['A_M'],
                 bins=25,
                 kde=True)
    plt.title('Absolute Magnitude for star Type : ' +
              star_type_names_df[index])
    plt.xlabel('Absolute Magnitude')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)

    plt.show()

stars_ohe = pd.get_dummies(
    data=stars, columns=['Color', 'Spectral_Class'], drop_first=True)
stars_ohe

sns.heatmap(stars_ohe.corr(),
            cmap='Blues')

# Visualizing correlation coefficients between features and target variable:
fig = plt.figure(figsize=(8, 10))
ax = sns.heatmap(stars_ohe.corr()[['Type']].sort_values(
    'Type', ascending=False), annot=True, annot_kws={"size": 12}, cmap='plasma')
ax.set_title(
    'Correlation Coefficient Between Each Feature and Star Type', fontsize=18)
ax.set_xlabel('Features', fontsize=16)
ax.set_ylabel('Features', fontsize=16)
# ax.tick_params(axis = "both", labelsize = 12)

stars_ohe.to_csv('FinalProcessed.csv')
