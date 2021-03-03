import pandas as pd
import yfinance as yf
import morningstar as ms
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
#kr = gm.KeyRatiosDownloader()
#kr_frames = kr.download('NDAQ')

#from morningstar.morningstar_client import MorningstarClient
#client = MorningstarClient()
#data = client.get_instrument_prices(instrument='28.10.F00000JQA9', start_date='01-01-2019', end_date='02-01-2019')

tickers = "MSFT AAPL CRM GOOGL TWTR NDAQ FB AA NFLX DIS GE BAC F"
data = [yf.download(ticker, start="2019-01-01", end="2019-04-30") for ticker in tickers.split(" ")]
#data.columns = ['{}_{}'.format(x[0], x[1]) for x in data.columns]
print(data[0].keys())

A = [D[D.columns.drop('Volume')].values for D in data]
mean_feature = np.stack([np.nan_to_num(np.mean(a, axis=0)) for a in A])
std_feature = np.stack([np.nan_to_num(np.std(a, axis=0)) for a in A])
X = np.hstack((mean_feature, std_feature))

def KMEANS(X):
    kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
    labels = kmeans.labels_
    return labels
def GMM(X):
    gm = GaussianMixture(n_components=2, random_state=0)
    labels = gm.fit_predict(X)
    print("Means are: {}".format(gm.means_))
    return labels

labels = GMM(X)
colors = ["r", "b", "g", "c"]
ticker_list = tickers.split(" ")
for i in range(max(labels) +1):
    cluster_i = X[labels == i]
    plt.scatter(cluster_i[:, 0], cluster_i[:, 1], color=colors[i], label="Cluster {}".format(i))
for i in range(len(ticker_list)):
    plt.annotate(ticker_list[i], (X[i,0],X[i,1]))
plt.legend()
plt.xlabel("Mean")
plt.ylabel("Variance")
plt.title("K-Means with Moment Features")
plt.show()


####### PCA ##########################
pca = PCA(n_components=2)
# New last line
X_dim_reduce = pca.fit_transform(X)

colors = ["r", "b", "g", "c"]
ticker_list = tickers.split(" ")
for i in range(max(labels) +1):
    cluster_i = X_dim_reduce[labels == i]
    plt.scatter(cluster_i[:, 0], cluster_i[:, 1], color=colors[i], label="Cluster {}".format(i))
for i in range(len(ticker_list)):
    plt.annotate(ticker_list[i], (X_dim_reduce[i,0],X_dim_reduce[i,1]))
plt.legend()
plt.xlabel("Mean")
plt.ylabel("Variance")
plt.title("K-Means with Moment Features")
plt.show()

############## Guassin Mixture ##########################




################## Hack Clustering #######################