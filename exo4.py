import matplotlib as mpl
mpl.use('TKAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.spatial import ConvexHull
from sklearn.mixture import GaussianMixture
from scipy import linalg
from sklearn.neighbors import NearestNeighbors

from sklearn import (manifold, decomposition, ensemble,
                     discriminant_analysis, random_projection)
					 
def convexHulls(points, labels):
    # computing convex hulls for a set of points with asscoiated labels
    convex_hulls = []
    for i in range(2):
        convex_hulls.append(ConvexHull(points[labels==i,:]))
    return convex_hulls
	
def best_ellipses(points, labels):
    # computing best fiiting ellipse for a set of points with asscoiated labels
    gaussians = []
    for i in range(2):
        gaussians.append(GaussianMixture(n_components=1, covariance_type='full').fit(points[labels==i, :]))
    return gaussians
	
def neighboring_hit(points, labels):
    k = 6
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(points)
    distances, indices = nbrs.kneighbors(points)

    txs = 0.0
    txsc = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    nppts = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    for i in range(len(points)):
        tx = 0.0
        for j in range(1,k+1):
            if (labels[indices[i,j]]== labels[i]):
                tx += 1
        tx /= k
        txsc[labels[i]] += tx
        nppts[labels[i]] += 1
        txs += tx

    for i in range(2):
        txsc[i] /= nppts[i]

    return txs / len(points)
	
def visualization(points2D, labels, convex_hulls, ellipses ,projname, nh):

    points2D_c= []
    for i in range(2):
        points2D_c.append(points2D[labels==i, :])
    # Data Visualization
    cmap =cm.tab10

    plt.figure(figsize=(3.841, 7.195), dpi=100)
    plt.set_cmap(cmap)
    plt.subplots_adjust(hspace=0.4 )
    plt.subplot(311)
    plt.scatter(points2D[:,0], points2D[:,1], c=labels,  s=3,edgecolors='none', cmap=cmap, alpha=1.0)
    plt.colorbar(ticks=range(2))

    plt.title("2D "+projname+" - NH="+str(nh*100.0))

    vals = [ i/2.0 for i in range(2)]
    sp2 = plt.subplot(312)
    for i in range(2):
        ch = np.append(convex_hulls[i].vertices,convex_hulls[i].vertices[0])
        sp2.plot(points2D_c[i][ch, 0], points2D_c[i][ch, 1], '-',label='$%i$'%i, color=cmap(vals[i]))
    plt.colorbar(ticks=range(2))
    plt.title(projname+" Convex Hulls")
	
## T-SNE
#tsne = manifold.TSNE(n_components=2, init='pca', random_state=0,perplexity=30,verbose = 2)
#X_tsne = tsne.fit_transform(X_test[:1000])

# PCA
#pca = decomposition.PCA(2)
#X_pca = pca.fit_transform(X_test[:1000])

# T-SNE
#convex_hulls_tsne = convexHulls(X_tsne, y_test[:1000])
#ellipses_tsne = best_ellipses(X_tsne, y_test[:1000])
#nh_tsne = neighboring_hit(X_tsne, y_test[:1000])


## Visualization for PCA
#visualization(X_pca, y_test[:1000], convex_hulls = convex_hulls_pca, ellipses = ellipses_pca,projname = 't-SNE', nh = nh_pca)



