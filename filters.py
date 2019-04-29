#!/usr/bin/python
# -*- coding: utf-8 -*-


import sys
import numpy as np
import sklearn.neighbors
from scipy import spatial
from numba import njit, autojit
#import xarray as xr


def get_radiusneighbors(radius):
    assert(radius < 0.2)  # evite que tout plante...
    return sklearn.neighbors.RadiusNeighborsRegressor(radius=radius)  # 5 cm de rayon


@autojit
def get_xy_z_data(data):
    if isinstance(data, np.ndarray):
        xydata = data[:, 0:2]  # coordonnees dans le plan
        zdata = data[:, 2]
    else:
        xydata = np.array((data[0], data[1])).T  # coordonnees dans le plan
        zdata = data[2]
    return xydata, zdata


#def radiusneighbors_generator(data,radius):
#
#    neigh = get_radiusneighbors(radius) # cherche les points dans un rayon de radius
#
#    n=100
#    i=0
#    while i<len(data):
#        neigh.fit(data,data[i:i+n]) # cherche les voisins
#        m = neigh.radius_neighbors_graph(data) # matrice de connectivity (contient des 1 entre les points voisins
#        yield i,m
#        i+=n


# trop lent!!
#def radiusneighbors_generator_spatial(data,radius):
#
#    tree = spatial.KDTree(data)
#
#    n=10000
#    i=0
#    while i<len(data):
#        tree2 = spatial.KDTree(data[i:i+n,:])
#        ivoisin = tree2.query_ball_tree(tree, radius)
#        yield i,ivoisin
#        i+=n

def radiusneighbors_generator(data, radius, return_distance=False, points=None, leaf_size=20):
    """for each points, return the nearest neighbors index list. If otherdata is given the query is performed on otherdata"""
    #m = sklearn.neighbors.radius_neighbors_graph(data, radius, mode='connectivity')
    #print 'start to generate'
    #for i in range(m.shape[0]):
    #    yield m.indices[m.indptr[i]:m.indptr[i+1]]

    #print 'create tree'
    tree = sklearn.neighbors.BallTree(data, leaf_size=leaf_size)

    if points is None:
        points = data

    n = 50000
    i = 0
    while i < len(points):
        #print 'search'
        res = tree.query_radius(points[i:i+n, :], radius, return_distance=return_distance)
        if return_distance:
            for ivoisin, dist in zip(*res):
                yield ivoisin, dist
        else:
            for ivoisin in res:
                yield ivoisin
        i += n


@autojit
def return_data(data, mask):
    if isinstance(data, np.ndarray):
        return data[mask, :].copy()
    else:
        return data[0][mask], data[1][mask], data[2][mask]


@autojit
def filter1(data):
    """filtre en cherchant les plus proches dans un rayon donné, et supprime ceux qui sont isolés. """

    if not isinstance(data, np.ndarray):
        data = np.array(data).T

    neigh = get_radiusneigbors(0.05)
    neigh.fit(data,data) 
    m = neigh.radius_neighbors_graph(data)

    mask = np.array((m.sum(axis=0) > 4)).squeeze()

    return return_data(data, mask)


@autojit
def filter2(data, radius=0.05, zmax=0.05):
    """ cherche les points dans le plan xy proche et vérifie qu'ils ne s'écartent pas trop en z de leur moyenne"""

    print('in filter2')

    xydata, zdata = get_xy_z_data(data)

    mask = np.empty_like(zdata, dtype=bool)
    for i, ind in enumerate(radiusneighbors_generator(xydata, radius)):
        mask[i] = (len(ind) > 3) & (abs(zdata[i] - zdata[ind].mean()) < zmax)  # moins de zmax de la moyenne et plus de 3 voisins

    #print mask.shape
    print('end filter2')
    return return_data(data, mask)


@autojit
def mean_lowerneighbors(zdata, dist, radius=0.05, rlower=0.3, nlower=5):
    """return the mean of the lowerpart of the set"""
    npoints = len(zdata)
    nl = int(max(nlower, npoints*rlower))  # rlower inferieur sinon nlower
    weights = (1-dist/radius)

    #if sum(weights)==0:
    #    print(len(weights), weights, dist, radius)

    if npoints > nl:
        ind = np.argpartition(zdata, nl)[0:nl]
        return np.average(zdata[ind], weights=weights[ind])
    else:
        return np.average(zdata, weights=weights)


@autojit
def mean_higherneighbors(zdata, dist, radius=0.05, rlower=0.3, nlower=5):
    """return the mean of the higherpart of the set"""
    npoints = len(zdata)
    nl = int(max(nlower, npoints*rlower))  # rlower inferieur sinon nlower
    weights = (1-dist/radius)

    #if sum(weights)==0:
    #    print(len(weights), weights, dist, radius)

    if npoints > nl:
        ind = np.argpartition(zdata, nl)[-nl:]
        return np.average(zdata[ind], weights=weights[ind])
    else:
        return np.average(zdata, weights=weights)


@autojit
def filter3(data, radius=0.10, rlower=0.3, nlower=5, zmax=0.2):
    """cherche les points dans le plan xy proche, puis calcule la moyenne de la partie basse. Retourne les points qui sont a 
    moins de zmax de cette partie base"""

    xydata, zdata = get_xy_z_data(data)

    npoints = len(zdata)
    minzdata = np.empty_like(zdata)

    mask = np.empty_like(zdata, dtype=bool)
    for i, (ivoisin, dist) in enumerate(radiusneighbors_generator(xydata, radius, return_distance=True)):

        zvoisin = zdata[ivoisin]  # get the neighbors
        mean_lower = mean_lowerneighbors(zvoisin, dist, radius=radius, rlower=rlower, nlower=nlower)  # compute the lower part

        #print(len(ivoisin), zdata[i], mean_lower)
        mask[i] = zdata[i] < mean_lower + zmax  # keep only those with z lower than mean_lower+zmax

    print('end filter3')
    return return_data(data, mask)


@autojit
def filter4(data, radius=0.10, zmax=0.05, r_threshold=0.3, n_threshold=1, leaf_size=20):
    """cherche les points dans le plan xy proche, et verifie qu'il y en a d'autres avec le meme z"""

    xydata, zdata = get_xy_z_data(data)

    mask = np.empty_like(zdata, dtype=bool)
    for i, ivoisin in enumerate(radiusneighbors_generator(xydata, radius, leaf_size=leaf_size)):

        zvoisin = zdata[ivoisin]  # get the neighbors
        n = np.sum(abs(zdata[i] - zvoisin) < zmax)  # number of point with a similar z

        mask[i] = n >= max(n_threshold, len(ivoisin)*r_threshold)

    #print('end filter4')
    return return_data(data, mask)


def mnt(data, radius=0.20, rlower=0.05, nlower=5):
    """cherche les points dans le plan xy proche, puis calcule la moyenne de la partie basse et attribue cette moyenne au point = Modele Numerique de Terrain"""

    xydata, zdata = get_xy_z_data(data)

    npoints = len(zdata)
    mntdata = np.empty_like(zdata)

    for i, (ivoisin, dist) in enumerate(radiusneighbors_generator(xydata, radius, return_distance=True)):

        zvoisin = zdata[ivoisin]  # get the neighbors
        mean_lower = mean_lowerneighbors(zvoisin, dist, radius=radius, rlower=rlower, nlower=nlower)  # compute the lower part

        mntdata[i] = mean_lower

    print('end mnt')

    if isinstance(data, np.ndarray):
        return np.hstack((xydata, mntdata[:, np.newaxis]))
    else:
        return data[0], data[1], mntdata


def mns(data, radius=0.20, rlower=0.05, nlower=5):
    """cherche les points dans le plan xy proche, puis calcule la moyenne de la partie haute et attribue cette moyenne au point: Modele Numerique de Surface"""

    xydata, zdata = get_xy_z_data(data)

    npoints = len(zdata)
    mnsdata = np.empty_like(zdata)

    for i, (ivoisin, dist) in enumerate(radiusneighbors_generator(xydata, radius, return_distance=True)):

        zvoisin = zdata[ivoisin]  # get the neighbors
        mean_higher = mean_higherneighbors(zvoisin, dist, radius=radius, rlower=rlower, nlower=nlower)  # compute the lower part

        mnsdata[i] = mean_higher

    print('end mns')

    if isinstance(data, np.ndarray):
        return np.hstack((xydata, mnsdata[:, np.newaxis]))
    else:
        return data[0], data[1], mnsdata


def filter_density(data, radius=0.10, n_points=3, density=None):
    """ne garde que les zones où il y a une certaine densité de points, ce qui indique que ca se passe bien dans cette zone"""

    xydata, zdata = get_xy_z_data(data)

    if density is not None:
        n_points = density / (np.pi*radius**2)

    mask = np.empty_like(zdata, dtype=bool)
    for i, ivoisin in enumerate(radiusneighbors_generator(xydata, radius)):
        mask[i] = len(ivoisin) > n_points

    #print('end filter_intensity')
    return return_data(data, mask)


def filter_intensity(data, radius=0.10, ratio_intensity=0.7):
    """cherche les points dans le plan xy proche, puis calcule la moyenne de l'intensité. Supprime ceux qui ont une intensité trop basse"""

    xydata, zdata = get_xy_z_data(data)

    idata = data[:, 3]  # assume we have an array, other cases to be implemented

    npoints = len(zdata)
    minzdata = np.empty_like(zdata)

    mask = np.empty_like(zdata, dtype=bool)
    for i, ivoisin in enumerate(radiusneighbors_generator(xydata, radius)):

        mean_intensity = np.mean(idata[ivoisin])  # get the neighbors

        #print(len(ivoisin), zdata[i], mean_lower)
        mask[i] = idata[i] > mean_intensity * ratio_intensity  # keep only those with an intensity not too low compared to the mean

    #print('end filter_intensity')
    return return_data(data, mask)


def make_grid(xydata, resolution):
    # make the grid
    grid = np.round(xydata/resolution)*resolution
    # remove duplicate:
    grid = np.array(list({(x, y) for (x, y) in grid}))  # probably not very efficient!

    return grid  # np.hstack((grid, np.zeros_like(grid)))


def thinning(data, resolution=0.005):
    """cherche les points dans chaque dx^2 et retourne un seul point"""

    xydata, zdata = get_xy_z_data(data)

    # make the grid
    grid = make_grid(xydata, resolution)

    mask = np.zeros_like(zdata, dtype=bool)
    #grid = mkgrid(xydata, resolution)
    for ivoisin in radiusneighbors_generator(xydata, resolution/2, points=grid):

        if len(ivoisin) == 1:
            mask[ivoisin] = True
        elif len(ivoisin) > 1:
            j = np.argmin(np.abs(zdata[ivoisin] - np.mean(zdata[ivoisin])))  # closest to the mean
            # print(len(ivoisin), j)
            mask[ivoisin[j]] = True

    # print('end filter_intensity')
    return return_data(data, mask)


def lowerpart(data, radius=0.1, nlower=5):
    """cherche les points dans le plan xy proche et prend les nlower ou un
       tiers si beaucoup de points les plus bas pour calculer la moyenne"""

    xydata, zdata = get_xy_z_data(data)

    npoints = len(zdata)
    minzdata = np.empty_like(zdata)

    j = 0
    for ivoisin, dist in radiusneighbors_generator(xydata,
                                                   radius,
                                                   return_distance=True):
        if j % 100000 == 0:
            print(j)
        zvoisin = zdata[ivoisin]

        minzdata[j] = mean_lowerneighbors(zvoisin, dist, radius)
        j += 1

    if isinstance(data, np.ndarray):
        return np.hstack((xydata, minzdata[:, np.newaxis]))
    else:
        return data[0], data[1], minzdata


def smooth(data, radius=0.05):
    """cherche les points dans le plan xy proche et prend la moyenne"""

    xydata, zdata = get_xy_z_data(data)

    meanzdata = np.empty_like(zdata)
    npoints = len(zdata)

    for i, (ivoisin, dist) in enumerate(radiusneighbors_generator(xydata, radius, return_distance=True)):
        if i % 100000 == 0:
            print(i)
        zvoisin = zdata[ivoisin]
        meanzdata[i] = np.average(zvoisin, weights=1-dist/radius)

    if isinstance(data, np.ndarray):
        return np.hstack((xydata, meanzdata[:, np.newaxis]))
    else:
        return data[0], data[1], meanzdata


def neighbors(data, radius=0.05):

    neigh = get_radiusneigbors(radius) # cherche les points dans un rayon de radius cm

    xydata = data[:, 0:2]  # coordonnees dans le plan

    neigh.fit(xydata, xydata)  # cherche les voisins
    m = neigh.radius_neighbors_graph(xydata)  # matrice de connectivity
    #npoints = np.array(m.sum(axis=0)).squeeze() # nombre de voisin
    npoints = np.array(m.sum(axis=1))  # nombre de voisin

    print(npoints.shape, data.shape)
    np.savetxt(open('ess.xyz', 'w'), np.concatenate((data, npoints), axis=1))


@autojit
def remove_outliers(zdata, threshold=5.0):

    m = np.nanmean(zdata)
    s = np.nanstd(zdata)
    zdata[zdata > m + threshold * s] = np.nan


@autojit
def concentric_mask(radius_min, radius_max, xydata):

    r2 = xydata[:, 0]**2 + xydata[:, 1]**2

    return np.logical_and(r2 > radius_min**2, r2 < radius_max**2)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(prog='filter')
    parser.add_argument('-l', '--location')
    parser.add_argument('filenames', nargs='*')
    args = parser.parse_args(sys.argv[1:])

    data = None
    for filename in args.filenames:
        data0 = np.loadtxt(filename, comments='//', skiprows=2)
        if data is None:
            data = data0
        else:
            print(data.shape)
            data = np.vstack((data, data0))
    data = mns(data)
    #data = lowerpart(data)
    #data = smooth(data,radius=0.1)
    print(data.shape)
    #np.savetxt(open('ground-lowerpart5-smooth10.xyz','w'),data.T)
    np.savetxt(open('_mns.xyz', 'w'), data)
