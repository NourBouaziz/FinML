# code ran on colab for faster execution and better performance
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from math import *

# loading dataset and preprocessing
stocks = pd.read_csv(r'preprocessed_CAC40.csv')
stocks.insert(loc=5, column='Daily percentage price movement', value=None, allow_duplicates=True)

for i in stocks.index:
    stocks['Daily percentage price movement'][i] = (stocks['Closing_Price'][i] - stocks['Open'][i]) / stocks['Open'][i]

# eliminating volume
stocks_to_calculate = pd.DataFrame(stocks, columns=['Name', 'Open', 'Closing_Price', 'Daily percentage price movement',
                                                    'Daily_High', 'Daily_Low'])
stocks_to_calculate.dropna(
    subset=['Open', 'Closing_Price', 'Daily percentage price movement', 'Daily_High', 'Daily_Low'], inplace=True)

# scaling
features = ['Open', 'Closing_Price', 'Daily percentage price movement', 'Daily_High', 'Daily_Low']
# Separating out the features
x = stocks_to_calculate.loc[:, features].values
# Separating out the target
y = stocks_to_calculate.loc[:, ['Name']].values
# Standardizing the features
x = StandardScaler().fit_transform(x)

# pca
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data=principalComponents
                           , columns=['principal component 1', 'principal component 2', 'principal component 3'])
finalDf = pd.concat([principalDf, stocks_to_calculate[['Name']]], axis=1)
Df_Construction = pd.concat([principalDf, stocks_to_calculate[['Name', 'Daily percentage price movement']]], axis=1)

finalDf.dropna(subset=['principal component 1', 'principal component 2', 'principal component 3'])
Df_Construction.dropna(subset=['principal component 1', 'principal component 2', 'principal component 3',
                               'Daily percentage price movement'])

# scattering examples of 5 stocks: arcelor, atos, credit agricole, dessault, essilor
fig1 = plt.figure(1)
ax = plt.axes(projection='3d')
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
ax.set_zlabel('Principal Component 3', fontsize=15)
ax.set_title('3 component PCA', fontsize=20)
names = ['ArcelorMittal', 'Atos', 'Crédit Agricole', 'Dassault Systèmes', 'EssilorLuxottica']
colors = ['red', 'green', 'blue', 'pink', 'black']
for name, color in zip(names, colors):
    indicesToKeep = finalDf['Name'] == name
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , finalDf.loc[indicesToKeep, 'principal component 3']
               , c=color
               , s=50)
ax.legend(names)
ax.grid()

# K-Means elbow curve testing
val = principalDf.values
sse = []
for k in range(2, 15):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(val)

    sse.append(kmeans.inertia_)
fig2 = plt.figure(2)
plt.plot(range(2, 15), sse)
plt.title("Elbow Curve")
plt.show()

# K-means actual clustering
kmeans = KMeans(n_clusters=6).fit(val)
centroids = kmeans.cluster_centers_
fig3 = plt.figure(3)
plt.scatter(val[:, 0], val[:, 1], val[:, 2], c=kmeans.labels_, cmap="rainbow")
plt.show()

# initializing 6 dataframes for the clusters
df_cluster0 = pd.DataFrame()
df_cluster1 = pd.DataFrame()
df_cluster2 = pd.DataFrame()
df_cluster3 = pd.DataFrame()
df_cluster4 = pd.DataFrame()
df_cluster5 = pd.DataFrame()

# filling the dataframes
i = -1
for val in kmeans.labels_:
    i += 1
    a_row = pd.Series([Df_Construction['Name'][i], Df_Construction['Daily percentage price movement'][i]])
    row_df = pd.DataFrame([a_row])
    if val == 0:
        df_cluster0 = pd.concat([row_df, df_cluster0], ignore_index=True)
    elif val == 1:
        df_cluster1 = pd.concat([row_df, df_cluster1], ignore_index=True)
    elif val == 2:
        df_cluster2 = pd.concat([row_df, df_cluster2], ignore_index=True)
    elif val == 3:
        df_cluster3 = pd.concat([row_df, df_cluster3], ignore_index=True)
    elif val == 4:
        df_cluster4 = pd.concat([row_df, df_cluster4], ignore_index=True)
    elif val == 5:
        df_cluster5 = pd.concat([row_df, df_cluster5], ignore_index=True)

# creating means, std dataframes for each cluster
df_means0 = pd.DataFrame(df_cluster0.groupby([0]).mean())
df_means1 = pd.DataFrame(df_cluster1.groupby([0]).mean())
df_means2 = pd.DataFrame(df_cluster2.groupby([0]).mean())
df_means3 = pd.DataFrame(df_cluster3.groupby([0]).mean())
df_means4 = pd.DataFrame(df_cluster4.groupby([0]).mean())
df_means5 = pd.DataFrame(df_cluster5.groupby([0]).mean())

df_std0 = pd.DataFrame(df_cluster0.groupby([0]).std())
df_std1 = pd.DataFrame(df_cluster1.groupby([0]).std())
df_std2 = pd.DataFrame(df_cluster2.groupby([0]).std())
df_std3 = pd.DataFrame(df_cluster3.groupby([0]).std())
df_std4 = pd.DataFrame(df_cluster4.groupby([0]).std())
df_std5 = pd.DataFrame(df_cluster5.groupby([0]).std())

df_means = [df_means0, df_means1, df_means2, df_means3, df_means4, df_means5]
df_std = [df_std0, df_std1, df_std2, df_std3, df_std4, df_std5]

for df_m, df_s in zip(df_means, df_std):
    df_m = df_m.rename(columns={0: "Name", 1: "Mean"})
    df_s = df_s.rename(columns={0: "Name", 1: "STD"})

# creating sharpe dataframe from means and std concatenation
df_sharpe0 = pd.concat([df_means0, df_std0], axis=1)
df_sharpe1 = pd.concat([df_means1, df_std1], axis=1)
df_sharpe2 = pd.concat([df_means2, df_std2], axis=1)
df_sharpe3 = pd.concat([df_means3, df_std3], axis=1)
df_sharpe4 = pd.concat([df_means4, df_std4], axis=1)
df_sharpe5 = pd.concat([df_means5, df_std5], axis=1)

# adding and calculating sharpe ratio
clusters = [df_sharpe0, df_sharpe1, df_sharpe2, df_sharpe3, df_sharpe4, df_sharpe5]
for cluster in clusters:
    cluster.insert(loc=2, column='Sharpe', value=None, allow_duplicates=True)
    for i in cluster.index:
        cluster['Sharpe'][i] = cluster['Mean'][i] / cluster['STD'][i]

# calculating percentage of stocks to extract from each cluster
n = []
for cluster in clusters:
    n.append(ceil(10 * len(cluster) / 155))

# sorting lines by sharpe ratio
for cluster in clusters:
    cluster = cluster.sort_values(by=['Sharpe'], ascending=False)

# we take the stocks with the highest sharpe ratios from each cluster
portfolio = []
l = -1
for cluster in clusters:
    l = l + 1
    i = 0
    for j in cluster.index:
        if i < n[l]:
            if j not in portfolio:
                i = i + 1
                portfolio.append(j)

print('Le portfeuille se constitue de:', portfolio)



