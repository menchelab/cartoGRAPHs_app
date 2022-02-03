#print('CSDEBUG: got to app_main')

########################################################################################
#
#  L I B R A R I E S
#
########################################################################################

from base64 import b64encode
import base64

import collections
from collections import defaultdict as dd
from collections import Counter as ct
from collections import OrderedDict
import colorsys
from colormath.color_objects import sRGBColor, LabColor
import csv

import dash_table
from dash.exceptions import PreventUpdate
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_table
import dash_bootstrap_components as dbc


import flask
from flask import Flask
from flask import send_file

import io
import itertools as it

import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import networkx as nx
from networkx.algorithms.flow import shortest_augmenting_path
from networkx.generators.degree_seq import expected_degree_graph
from networkx.algorithms.community import greedy_modularity_communities
from networkx.readwrite.adjlist import parse_adjlist
from networkx.readwrite.edgelist import parse_edgelist
import networkxgmml as nxml

import numpy as np
from numpy import gradient
from numpy import pi, cos, sin, arccos, arange
import numpy.linalg as la

import os
import os.path

import pandas as pd
import pickle
import plotly
import plotly.graph_objs as pgo
import plotly.offline as py
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, iplot
import plotly.io as pio
import pylab
from pkg_resources import get_default_cache
from plotly.missing_ipywidgets import FigureWidget
from plotly.io import to_image

import random as rd

from scipy.spatial import distance
from scipy.cluster.hierarchy import fcluster
import scipy.stats as st
from scipy import stats

import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as dist
from scipy.interpolate import interpn
from scipy.stats import gaussian_kde

import seaborn as sns

import sklearn
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn import datasets
from sklearn.preprocessing import normalize
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection,cluster)
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import LinearRegression

import statistics
import sys


import time

import umap
from urllib.parse import quote as urlquote
import urllib

import warnings

#print('CSDEBUG: app_main imports complete')
from numba import config, njit, threading_layer
#print('CSDEBUG: numba imports successful')
# set the threading layer before any parallel target compilation
config.THREADING_LAYER = 'omp'
#print('threading layer set')

@njit('float64(float64[::1], float64[::1])')
def foo(a, b):
    return a[1] + b[2]
#print('COMPILED OK')

x = np.arange(10.)
y = x.copy()

# # this will force the compilation of the function, select a threading layer
# # and then execute in parallel
#print(foo(x, y))
#print('EXECUTED OK')
#print('CSDEBUG: function compilation successful')
# demonstrate the threading layer chosen
#print("Threading layer chosen: %s" % threading_layer())

########################################################################################
#
# F U N C T I O N S   T O  L O A D  D A T A
#
########################################################################################


############################
#
# Data Parsing Functions
#
############################

def parse_Graph(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            G = nx.read_edgelist(io.StringIO(decoded.decode('utf-8')), delimiter=',')
        elif 'txt' in filename:
            G = nx.read_edgelist(io.BytesIO(decoded))

    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return G



def parse_Featurematrix(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            FM = pd.read_csv(io.StringIO(decoded.decode('utf-8')), index_col=0)
        elif 'xls' in filename:
            FM = pd.read_excel(io.BytesIO(decoded), index_col=0)
    except Exception as e:
        return html.Div([
            'There was an error processing this file.'
        ])
    return FM


def import_vrnetzer_csv(G,file):

    edge_width = 0.8
    edge_opac = 0.05
    edge_colordark = '#666666'
    node_size = 1.5
    #nodesglow_diameter = 20.0
    #nodesglow_transparency = 0.05 # 0.01

    df = pd.read_csv(file, header=None)

    df.columns = ['id','x','y','z','r','g','b','a','namespace']

    df_vrnetzer = df.set_index('id')
    df_vrnetzer.index.name = None
    #print(df_vrnetzer)

    ids = [str(i) for i in list(df['id'])]
    x = list(df['x'])
    y = list(df['y'])
    z = list(df['z'])
    posG = dict(zip(ids,zip(x,y,z)))

    r_list = list(df['r'])
    g_list = list(df['g'])
    b_list = list(df['b'])
    a_list = list(df['a'])

    colours = list(zip(r_list,g_list,b_list,a_list))

    umap_nodes = get_trace_nodes_3D(posG, ids , colours, node_size)# size3d)
    umap_edges = get_trace_edges_3D(G, posG, edge_colordark, edge_opac, edge_width)
    umap_data= [umap_edges, umap_nodes]
    fig = plot3D_app(umap_data)

    return fig, df_vrnetzer


def load_graph(organism):

    if organism == 'yeast':

        data = pickle.load( open("input/BIOGRID-ORGANISM-Saccharomyces_cerevisiae_S288c-3.5.185.mitab.pickle", "rb" ) )

        filter_score = data[
                            #(data['Interaction Types'] == 'psi-mi:"MI:0915"(physical association)') +
                            (data['Interaction Types'] == 'psi-mi:"MI:0407"(direct interaction)')
                            #&
                            #(data['Taxid Interactor A'] == "taxid:559292") &
                            #(data['Taxid Interactor B'] == "taxid:559292")
        ]

        g = nx.from_pandas_edgelist(filter_score, '#ID Interactor A', 'ID Interactor B')
        g.remove_edges_from(nx.selfloop_edges(g)) #remove self loop

        G_cere = g.subgraph(max(nx.connected_components(g), key=len)) # largest connected component (lcc)
        G = G_cere

        return G

    elif organism == 'human':

        G = nx.read_edgelist('input/ppi_elist.txt',data=False)
        return G

    else:
        print('Please choose organism by typing "human" or "yeast"')


def load_genesymbols(G,organism):
    '''
    Load prepared symbols of genes.
    Input:
    - organism = string; choose from 'human' or 'yeast'

    Return dictionary of geneID (keys) and symbols (values).
    '''
    if organism == 'yeast':
        df_gID_sym = pd.read_csv('input/DF_gene_symbol_yeast.csv', index_col=0)
        gene_sym = list(df_gID_sym['Sym'])
        gene_id = list(df_gID_sym.index)
        d_gene_sym  = dict(list(zip(gene_id, gene_sym)))

        return d_gene_sym

    elif organism == 'human':
        df_gene_sym = pd.read_csv('input/DF_gene_symbol_human.csv', index_col=0)
        sym = list(df_gene_sym['0'])
        l_features = []
        for i in sym:
            l_features.append(i[2:-2])
        d_gene_sym = dict(zip(G.nodes(),l_features))

        return d_gene_sym

    else:
        print('Please choose organism by typing "human" or "yeast"')



def load_centralities(G,organism):
        '''
        Load prepared centralities of genes.
        Input:
        - G = Graph
        - organism = string; choose from 'human' or 'yeast'

        Return dictionary with genes as keys and four centrality metrics as values.
        '''
        df_centralities = pd.read_csv('input/Features_centralities_Dataframe_'+organism+'.csv', index_col=0)

        d_deghubs = dict(G.degree())
        d_clos = dict(zip(G.nodes(), df_centralities['clos']))
        d_betw = dict(zip(G.nodes(), df_centralities['betw']))
        d_eigen = dict(zip(G.nodes(), df_centralities['eigen']))

        d_centralities = dict(zip(list(G.nodes),zip(d_deghubs.values(),d_clos.values(),d_betw.values(),d_eigen.values())))

        #cent_features = []
        #for i in d_centralities.items():
        #    k=list(i)
        #    cent_features.append(k)

        return d_centralities



def load_essentiality(G, organism):
        '''
        Load prepared essentiality state of organism.
        Input:
        - organism = string; choose from 'human' or 'yeast'

        Return lists of genes, split based on essentiality state.
        '''
        if organism == 'human':

            # ESSENTIALITY
            # get dataframe with ENSG-ID and essentiality state
            df_human_ess = pd.read_table("input/human_essentiality.txt", delim_whitespace=True)

            # create dict with ENSG-ID:essentiality state
            ensg_id = list(set(df_human_ess['sciName']))
            gene_ess = list(df_human_ess['locus'])
            d_ensg_ess = dict(zip(ensg_id, gene_ess))

            # match ENSG-ID with entrezID
            # "engs_to_entrezid": entrezIDs were matched with "ensg_id.txt" via "DAVID Database" (https://david.ncifcrf.gov/conversion.jsp)
            df_human_ensg_entrez = pd.read_table('input/ensg_to_entrezid.txt') # delim_whitespace=False)
            df_human_ensg_entrez.dropna()

            df = df_human_ensg_entrez
            df['To'] = df['To'].fillna(0)
            df['To'] = df['To'].astype(int)
            df_human_ensg_entrez = df

            # create dict with ENGS-ID: entrezID
            ensgid = list(df_human_ensg_entrez['From']) #engs ID
            entrezid = list(df_human_ensg_entrez['To']) #entrez ID

            # dict with engsid : entrezid
            d_ensg_entrez = dict(zip(ensgid, entrezid))

            # create dict with entrezID:essentiality state
            d_id_ess_unsorted = {}
            for ens,ent in d_ensg_entrez.items():
                for en, ess in d_ensg_ess.items():
                    if ens == en:
                        d_id_ess_unsorted[str(ent)] = ess


            # check if G.nodes match entrezID in dict and sort according to G.nodes
            d_gid_ess = {}
            for k,v in d_id_ess_unsorted.items():
                if k in G.nodes():
                    d_gid_ess[k]=v

            # create dict with rest of G.nodes not in dict (entrezID:essentiality)
            d_gid_rest = {}
            for g in G.nodes():
                if g not in d_gid_ess.keys():
                    d_gid_rest[g]='not defined'

            #print(len(d_gid_rest)+len(d_gid_ess)) # this should match G.nodes count

            # merge both dicts
            d_gid_ess_all_unsorted = {**d_gid_ess, **d_gid_rest}

            # sort -> G.nodes()
            d_gID_all = {key:d_gid_ess_all_unsorted[key] for key in G.nodes()}

            essential_genes = []
            non_ess_genes = []
            notdefined_genes = []
            for k,v in d_gID_all.items():
                if v == 'E':
                    essential_genes.append(k)
                elif v == 'NE':
                    non_ess_genes.append(k)
                else:
                    notdefined_genes.append(k)

            return essential_genes,non_ess_genes,notdefined_genes


        elif organism == 'yeast':

            # ESSENTIALITY
            cere_gene =pd.read_csv("input/Saccharomyces cerevisiae.csv",
                       delimiter= ',',
                       skipinitialspace=True)

            cere_sym = list(cere_gene['symbols'])
            cere_ess = list(cere_gene['essentiality status'])
            cere_sym_essentiality = dict(zip(cere_sym, cere_ess))

            d_cere_ess = {}
            d_cere_noess = {}
            d_cere_unknown = {}

            for node,es in cere_sym_essentiality.items():
                if es == 'E':
                    d_cere_ess[node]=es
                elif es == 'NE':
                    d_cere_noess[node]=es

            d_cere_alless = {}
            for nid, sym in g_ID_sym.items():
                for sy,ess in cere_sym_essentiality.items():
                    if sym == sy:
                        d_cere_alless[nid] = ess

            d_cere_unknown = {}
            for g in G.nodes():
                if g not in d_cere_alless.keys():
                    d_cere_unknown[g]='status unkonwn'

            d_geneID_ess = {**d_cere_unknown, **d_cere_alless}

            d_gID_ess = {}
            d_gID_noess = {}
            d_gID_notdef = {}

            for k,i in d_geneID_ess.items():
                if i == 'E':
                    d_gID_ess[k] = i
                elif i == 'NE':
                    d_gID_noess[k] = i
                else:
                    d_gID_notdef[k] = 'not defined'

            d_gID_all_unsorted = {**d_gID_ess, **d_gID_noess, **d_gID_notdef}
            d_gID_all = {key:d_gID_all_unsorted[key] for key in G.nodes()}

            essential_genes = []
            non_ess_genes = []
            notdefined_genes = []
            for k,v in d_gID_all.items():
                if v == 'E':
                    essential_genes.append(k)
                elif v == 'NE':
                    non_ess_genes.append(k)
                else:
                    notdefined_genes.append(k)

            return essential_genes,non_ess_genes,notdefined_genes

        else:
            print('Please choose organism by typing "human" or "yeast"')



def load_structural_datamatrix(G,organism,netlayout):
    '''
    Load precalculated Matrix with N genes and M features.
    Input:
    - path = directory of file location
    - organism = string; choose from 'human' or 'yeast'
    - netlayout = string; choose a network layout e.g. 'local', 'global', 'importance', 'funct-bio', 'funct-cel', 'funct-mol', funct-dis'

    Return Matrix based on choice.
    '''
    path = 'input/'

    if netlayout == 'local':

        DM_adj = pd.read_csv(path+'Adjacency_Dataframe_'+organism+'.csv', index_col=0)
        DM_adj.index = list(G.nodes())
        DM_adj.columns = list(G.nodes())

        return DM_adj

    elif netlayout == 'global':

        #r = 0.9
        #alpha = 0.9-1.0

        DM_m_visprob_transposed = pd.read_csv(path+'RWR_Dataframe_'+organism+'.csv', index_col=0)
        DM_m_visprob_transposed.index = list(G.nodes())
        DM_m_visprob_transposed.columns = list(G.nodes())

        return DM_m_visprob_transposed

    elif netlayout == 'importance':

        df_centralities = load_centralities(organism)
        DM_centralities = pd.DataFrame(distance.squareform(distance.pdist(df_centralities, 'cosine')))

        DM_centralities = round(DM_centralities,6)
        DM_centralities.index = list(G.nodes())
        DM_centralities.columns = list(G.nodes())

        return DM_centralities

    elif netlayout == 'funct-bio' and organism == 'human':

        DM_BP = pd.read_csv(path+'DistanceMatrix_goBP_Dataframe_human_cosine.csv', index_col=0)
        DM_BP_round = DM_BP.round(decimals=6)

        return DM_BP_round


    elif netlayout == 'funct-mol' and organism == 'human':

        DM_MF = pd.read_csv('input/DistanceMatrix_goMF_Dataframe_Human_cosine.csv', index_col=0)
        DM_MF_round = DM_MF.round(decimals=6)

        return DM_MF_round

    elif netlayout == 'funct-cel' and organism == 'human':

        DM_CC = pd.read_csv('input/DistanceMatrix_goCC_Dataframe_Human_cosine.csv', index_col=0)
        DM_CC_round = DM_CC.round(decimals=6)

        return DM_CC_round

    elif netlayout == 'funct-dis' and organism == 'human':

        DM_Disease = pd.read_csv('input/DistanceMatrix_Disease_Dataframe_Human_cosine.csv', index_col=0)
        DM_Disease_round= DM_Disease.round(decimals=6)

        return DM_Disease_round

    else:
        print('Please type one of the following: "local", "global", "importance", "functional"')


########################################################################################
#
# F U N C T I O N S   F O R   A N A L Y S I S + C A L C U L A T I O N S
#
########################################################################################


def compute_centralityfeatures(G):
    '''
    Compute degree,betweenness,closeness and eigenvector centrality
    Input: 
    - G: networkx Graph 
    
    Return a dictionary sorted according to G.nodes with nodeID as keys and four centrality values. 
    ''' 
    
    degs = dict(G.degree())
    d_deghubs = {}
    for node, de in sorted(degs.items(),key = lambda x: x[1], reverse = 1):
        d_deghubs[node] = round(float(de/max(degs.values())),4)

    closeness = nx.closeness_centrality(G)
    d_clos = {}
    for node, cl in sorted(closeness.items(), key = lambda x: x[1], reverse = 1):
        d_clos[node] = round(cl,4)

    betweens = nx.betweenness_centrality(G)
    d_betw = {}
    for node, be in sorted(betweens.items(), key = lambda x: x[1], reverse = 1):
         d_betw[node] = round(be,4)

    eigen = nx.eigenvector_centrality(G)
    d_eigen = {}
    for node, eig in sorted(eigen.items(), key = lambda x: x[1], reverse = 1):
         d_eigen[node] = round(eig,4)

    d_deghubs_sorted = {key:d_deghubs[key] for key in sorted(d_deghubs.keys())}
    d_clos_sorted = {key:d_clos[key] for key in sorted(d_clos.keys())}
    d_betw_sorted = {key:d_betw[key] for key in sorted(d_betw.keys())}
    d_eigen_sorted = {key:d_eigen[key] for key in sorted(d_eigen.keys())}

    feature_dict = dict(zip(d_deghubs_sorted.keys(), zip(d_deghubs_sorted.values(),d_clos_sorted.values(),d_betw_sorted.values(),d_eigen_sorted.values())))

    feature_dict_sorted = {key:feature_dict[key] for key in G.nodes()}
    
    return feature_dict_sorted



def rnd_walk_matrix2(A, r, a, num_nodes):
    '''
    Random Walk Operator with restart probability.
    Input: 
    - A = Adjanceny matrix (numpy array)
    - r = restart parameter e.g. 0.9
    - a = teleportation value e.g. 1.0 for max. teleportation
    - num_nodes = all nodes included in Adjacency matrix, e.g. amount of all nodes in the graph 

    Return Matrix with visiting probabilites (non-symmetric!!).
    ''' 
    n = num_nodes
    factor = float((1-a)/n)

    E = np.multiply(factor,np.ones([n,n]))              # prepare 2nd scaling term
    A_tele = np.multiply(a,A) + E  #     print(A_tele)
    M = normalize(A_tele, norm='l1', axis=0)                                 # column wise normalized MArkov matrix

    # mixture of Markov chains
    del A_tele
    del E

    U = np.identity(n,dtype=int) 
    H = (1-r)*M
    H1 = np.subtract(U,H)
    del U
    del M
    del H    

    W = r*np.linalg.inv(H1)   

    return W


def bin_nodes(data_dict): 
    '''
    Binning nodes based on unique values in dictionary input. 
    Input: 
    - data_dict = dictionary with node id as keys and values of e.g. degree centrality.
    
    Return binned nodes.
    '''
    bins = set(data_dict.values())

    d_binned = {}
    for n in bins:
        d_binned[n]=[str(k) for k in data_dict.keys() if data_dict[k] == n]
        
    return d_binned


def rotate_z(x, y, z, theta):
    '''
    Function to make 3D html plot rotating.
    Returns frames, to be used in "pgo.Figure(frames = frames)"
    '''
    
    w = x+1j*y
    return np.real(np.exp(1j*theta)*w), np.imag(np.exp(1j*theta)*w), z 


########################################################################################
#
# E M B E D D I N G + R E L A T E D
#
########################################################################################


# -------------------------------------------------------------------------------------
# A N N O T A T I O N S + L A B E L S 
# -------------------------------------------------------------------------------------

def annotation_kmeansclustering(kmeans, centrs, mode):
   
    # number of genes in each cluster ( used for annotation )
    d_clus_genecount = dict(collections.Counter(kmeans.labels_))
    d_clus_genecount_sort = dict(collections.OrderedDict(sorted(d_clus_genecount.items())))
    l_clus_genecount = list(d_clus_genecount_sort.values())

    # Centers for clusters ( used for annotation )
    x=[]
    y=[]
    z=[]
    for i in centrs:
        x.append(i[0])
        y.append(i[1])
        z.append(i[2])
        
    if mode == 'light':
        annotations = []
        for i in range(len(x)):
            annot = dict(
                                x=x[i],
                                y=y[i],
                                z=z[i],
                                showarrow=True,
                                text=f'Cluster: {str(i+1)} <br> total: {str(l_clus_genecount[i])}', 
                                font=dict(
                                    color="dimgrey",
                                    size=8),
                                xanchor="right",
                                ay=-20,
                                ax=-20,
                                opacity=0.5,
                                arrowhead=0,
                                arrowwidth=0.5,
                                arrowcolor="dimgrey"
                                )
            i=+1
            annotations.append(annot)
        return annotations

    elif mode == 'dark':
        annotations = []
        for i in range(len(x)):
            annot = dict(
                                x=x[i],
                                y=y[i],
                                z=z[i],
                                showarrow=True,
                                text=f'Cluster: {str(i+1)} <br> total: {str(l_clus_genecount[i])}',
                                font=dict(
                                    color="lightgrey",
                                    size=8),
                                xanchor="right",
                                ay=-20,
                                ax=-20,
                                opacity=0.5,
                                arrowhead=0,
                                arrowwidth=0.5,
                                arrowcolor="lightgrey")
            i=+1
            annotations.append(annot)
        return annotations
        
    else: 
        print('Please choose mode by setting mode="light" or "dark".')
        
        
        
def cluster_annotation(d_clusterid_coords, d_genes_per_cluster, mode = 'light'):
    ''' 
    Add Anntation of clusters to 3D plot.
    Input:
    - d_clusterid_coords = dictionary with cluster id and x,y,z coordinates of cluster center.
    - d_genes_per_cluster = dictionary with cluster id and genes counted per cluster 
    - mode = mode of plot (i.e. 'light', 'dark')
    
    Return Annotations for each cluster.
    '''    
    
    l_clus_genecount = list(d_genes_per_cluster.values())

    x=[]
    y=[]
    z=[]
    for i in d_clusterid_coords.values():
        x.append(i[0])
        y.append(i[1])
        z.append(i[2])

    if mode == 'light':
        annotations = []
        for i in range(len(x)):
            annot = dict(
                                x=x[i],
                                y=y[i],
                                z=z[i],
                                showarrow=True,
                                text=f'Cluster: {str(i+1)} <br> total: {str(l_clus_genecount[i])}', 
                                font=dict(
                                    color="dimgrey",
                                    size=8),
                                xanchor="right",
                                ay=-100,
                                ax=-100,
                                opacity=0.5,
                                arrowhead=0,
                                arrowwidth=0.5,
                                arrowcolor="dimgrey"
                                )
            i=+1
            annotations.append(annot)
        return annotations

    elif mode == 'dark':
        annotations = []
        for i in range(len(x)):
            annot = dict(
                                x=x[i],
                                y=y[i],
                                z=z[i],
                                showarrow=True,
                                text=f'Cluster: {str(i+1)} <br> total: {str(l_clus_genecount[i])}',
                                font=dict(
                                    color="lightgrey",
                                    size=8),
                                xanchor="right",
                                ay=-100,
                                ax=-100,
                                opacity=0.5,
                                arrowhead=0,
                                arrowwidth=0.5,
                                arrowcolor="lightgrey")
            i=+1
            annotations.append(annot)
        return annotations
        
    else: 
        print('Please choose mode by setting mode="light" or "dark".')

    

def genes_annotation(posG_genes, d_genes, mode = 'light'):
    '''
    Add Anntation of genes to 3D plot.
    Input:
    - posG_genes = dictionary with node id and x,y,z coordinates.
    - d_genes = dictionary with node id as keys and symbol (gene symbol) as values. Same order as posG_genes
    - mode of plot (i.e. 'light', 'dark')
    
    Return Annotations for each cluster.
    ''' 
    
    gene_sym = list(d_genes.values())
    
    x = []
    y = []
    z = []
    for k,v in posG_genes.items():
        x.append(v[0])
        y.append(v[1])
        z.append(v[2])

    if mode == 'light':
        annotations = []
        for i in range(len(x)):
            annot = dict(
                                    x=x[i],
                                    y=y[i],
                                    z=z[i],
                                    showarrow=True,
                                    text=f'Gene: {gene_sym[i]}',                            
                                    font=dict(
                                        color="dimgrey",
                                        size=10),
                                    xanchor="right",
                                    ay=-10,
                                    ax=-10,
                                    opacity=0.5,
                                    arrowhead=0,
                                    arrowwidth=0.5,
                                    arrowcolor="dimgrey")
            i=+1
            annotations.append(annot)
        return annotations

    elif mode == 'dark':
        annotations = []
        for i in range(len(x)):
            annot = dict(
                                    x=x[i],
                                    y=y[i],
                                    z=z[i],
                                    showarrow=True,
                                    text=f'Gene: {gene_sym[i]}',
                                    font=dict(
                                        color="white",
                                        size=10),
                                    xanchor="right",
                                    ay=-10,
                                    ax=-10,
                                    opacity=0.5,
                                    arrowhead=0,
                                    arrowwidth=0.5,
                                    arrowcolor="lightgrey")
            i=+1
            annotations.append(annot)
        return annotations

    else: 
        print('Please choose mode by setting mode="light" or "dark".')

        
        
def annotation_disease(position_annot, disease_names, disease_colors, mode):
   
    # number of genes in each cluster ( used for annotation )

    if mode == 'light':
        annotations = []
        for i in range(len(disease_names)):
            annot = dict(
                                x=position_annot[i][0],
                                y=position_annot[i][1],
                                z=position_annot[i][2],
                                showarrow=True,
                                text=disease_names[i], 
                                font=dict(
                                    color=disease_colors[i],
                                    size=14),
                                xanchor="right",
                                ay=0,
                                ax=0,
                                opacity=1,
                                arrowhead=0,
                                arrowwidth=0.5,
                                arrowcolor="dimgrey"
                                )
            i=+1
            annotations.append(annot)
        return annotations

    elif mode == 'dark':
        annotations = []
        for i in range(len(disease_names)):
            annot = dict(
                                x=position_annot[i][0],
                                y=position_annot[i][1],
                                z=position_annot[i][2],
                                showarrow=True,
                                text=disease_names[i],
                                font=dict(
                                    color=disease_colors[i],
                                    size=14),
                                xanchor="right",
                                ay=0,
                                ax=0,
                                opacity=1,
                                arrowhead=0,
                                arrowwidth=0.5,
                                arrowcolor="lightgrey")
            i=+1
            annotations.append(annot)
        return annotations
        
    else: 
        print('Please choose mode by setting mode="light" or "dark".')

        
########################################################################################

# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
#
#      ######     #######
#    ##     ##    ##    ##
#           ##    ##     ## 
#          ##     ##     ##
#        ##       ##     ##
#      ##         ##     ##
#    ##           ##    ##
#    ##########   #######
#
# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# E M B E D D I N G 
# -------------------------------------------------------------------------------------


def embed_tsne_2D(Matrix, prplxty, density, l_rate, steps, metric = 'precomputed'):
    '''
    Dimensionality reduction from Matrix using t-SNE.
    Return dict (keys: node IDs, values: x,y).
    ''' 
    
    tsne = TSNE(n_components = 2, random_state = 0, perplexity = prplxty, metric = metric, init='pca',
                     early_exaggeration = density,  learning_rate = l_rate ,n_iter = steps,
                     square_distances=True)
    
    embed = tsne.fit_transform(Matrix)
    
    return embed



def embed_umap_2D(Matrix, n_neigh, spre, m_dist, metric='cosine', learn_rate = 1, n_ep = None):
    '''
    Dimensionality reduction from Matrix using UMAP.
    Return dict (keys: node IDs, values: x,y).
    ''' 
    n_comp = 2 

    U = umap.UMAP(
        n_neighbors = n_neigh,
        spread = spre,
        min_dist = m_dist,
        n_components = n_comp,
        metric = metric, 
        random_state=42,
        learning_rate = learn_rate, 
        n_epochs = n_ep)
    
    embed = U.fit_transform(Matrix)
    
    return embed


def get_posG_2D(l_nodes, embed):
    '''
    Get 2D coordinates for each node.
    Return dict with node: x,y coordinates.
    '''
    
    posG = {}
    cc = 0
    for entz in l_nodes:
        posG[entz] = (embed[cc,0],embed[cc,1])
        cc += 1

    return posG



def get_posG_2D_norm(G, DM, embed, r_scalingfactor=1.2):
    '''
    Generate coordinates from embedding. 
    Input:
    - G = Graph
    - DM = matrix; index and columns must be same as G.nodes
    - embed = embedding from e.g. tSNE , UMAP ,... 
    
    Return dictionary with nodes as keys and coordinates as values in 3D normed. 
    ''' 
        
    genes = []
    for i in DM.index:
        if str(i) in G.nodes() or int(i) in G.nodes():
            genes.append(str(i))

    genes_rest = [] 
    for i in G.nodes():
        if str(i) not in genes:
            genes_rest.append(str(i))

    posG = {}
    cc = 0
    for entz in genes:
        posG[entz] = (embed[cc,0],embed[cc,1])
        cc += 1

    #--------------------------------------------------------------
    # REST (if genes = G.nodes then rest will be ignored / empty)
    
    # generate circle coordinates for rest genes (without e.g. GO term or Disease Annotation)
    t = np.random.uniform(0,2*np.pi,len(genes_rest))
    
    xx=[]
    yy=[]
    for i in posG.values():
        xx.append(i[0])
        yy.append(i[1])
    
    cx = np.mean(xx)
    cy = np.mean(yy)

    xm, ym = max(posG.values())
    r = (math.sqrt((xm-cx)**2 + (ym-cy)**2))*r_scalingfactor #*1.05 # multiplying with 1.05 makes cirle larger to avoid "outsider nodes/genes"
        
    x = r*np.cos(t)
    y = r*np.sin(t)
    rest = []
    for i,j in zip(x,y):
            rest.append((i,j))

    posG_rest = dict(zip(genes_rest, rest))

    posG_all = {**posG, **posG_rest}
    
    #G_nodes_str = [str(i) for i in list(G.nodes())]
    posG_complete = {key:posG_all[key] for key in list(G.nodes())}

    # normalize coordinates 
    x_list = []
    y_list = []
    for k,v in posG_complete.items():
        x_list.append(v[0])
        y_list.append(v[1])

    xx_norm = preprocessing.minmax_scale(x_list, feature_range=(0, 1), axis=0, copy=True)
    yy_norm = preprocessing.minmax_scale(y_list, feature_range=(0, 1), axis=0, copy=True)

    xx_norm_final=[]
    for i in xx_norm:
        xx_norm_final.append(round(i,10))

    yy_norm_final=[]
    for i in yy_norm:
        yy_norm_final.append(round(i,10))

    posG_complete_norm = dict(zip(list(G.nodes()),zip(xx_norm_final,yy_norm_final)))

    return posG_complete_norm




# -------------------------------------------------------------------------------------
# L A B E L S 
# -------------------------------------------------------------------------------------

def labels2D(posG, feature_dict):
    '''
    Create Node Labels, based on a dict of coordinates (keys:node ID, values: x,y)
    Return new dict of node iDs and features for each node.
    '''

    labels = {} 
    c = 0
    for node, xy in sorted(posG.items(), key = lambda x: x[1][0]):
        labels[node] = ([node,feature_dict[node][0],feature_dict[node][1],feature_dict[node][2],feature_dict[node][3]])   
        c+=1
        
    return labels


def position_labels(posG, move_x, move_y):
    '''
    Create label position of coordinates dict.
    Return new dict with label positions. 
    '''    
    
    posG_labels = {}
    for key,val in posG.items():
        xx = val[0] + move_x
        yy = val[1] + move_y
        posG_labels[key] = (xx,yy)
        
    return posG_labels


# -------------------------------------------------------------------------------------
# P L O T T I N G 
# -------------------------------------------------------------------------------------

def get_trace_nodes_2D(posG, info_list, color_list, size, linewidth=0.25, opac = 0.8):
    '''
    Get trace of nodes for plotting in 2D. 
    Input: 
    - G = Graph
    - posG = dictionary with nodes as keys and coordinates as values.
    - color_list = list of colors obtained from any color function (see above sections).
    - opacity = transparency of edges e.g. 0.2
    
    Return a trace for plotly graph objects plot. 
    '''
    
    key_list=list(posG.keys())
    trace = pgo.Scatter(x=[posG[key_list[i]][0] for i in range(len(key_list))],
                           y=[posG[key_list[i]][1] for i in range(len(key_list))],
                           mode = 'markers',
                           text = info_list,
                           hoverinfo = 'text',
                           #textposition='middle center',
                           marker = dict(
                color = color_list,
                size = size,
                symbol = 'circle',
                line = dict(width = linewidth,
                        color = 'dimgrey'),
                opacity = opac
            ),
        )
    
    return trace



def color_edges_from_nodelist_specific(G, l_nodes, color):
    '''
    Color (highlight) edges from specific node list exclusively.
    Input:
    - G = Graph 
    - l_nodes = list of nodes 
    - color = string; color to hightlight
    
    Return edge list for selected edges IF BOTH nodes are in l_nodes. 
    '''
    
    edge_lst = [(u,v)for u,v in G.edges(l_nodes) if u in l_nodes and v in l_nodes]

    d_col_edges = {}
    for e in edge_lst:
        d_col_edges[e]=color
    return d_col_edges



def get_trace_edges_2D(G, posG, color, opac = 0.2, linewidth = 0.2):
    '''
    Get trace of edges for plotting in 2D. 
    Input: 
    - G = Graph
    - posG = dictionary with nodes as keys and coordinates as values.
    - color = string; hex color
    - opacity = transparency of edges e.g. 0.2
    
    Return a trace for plotly graph objects plot. 
    '''
    
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = posG[edge[0]]
        x1, y1 = posG[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
 
    trace_edges = pgo.Scatter(
                        x = edge_x, 
                        y = edge_y, 
                        mode = 'lines', hoverinfo='none',
                        line = dict(width = linewidth, color = color),
                        opacity = opac
                )
    
    return trace_edges


def get_trace_edges_specific2D(d_edges_col, posG, linew = 0.75, opac=0.1):

    edge_x = []
    edge_y = []
    
    for edge, col in d_edges_col.items():
            x0, y0 = posG[edge[0]]
            x1, y1 = posG[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
            
    cols = list(d_edges_col.values())[0]
    
    trace_edges = pgo.Scatter(
                        x = edge_x, 
                        y = edge_y, 
                        mode = 'lines', hoverinfo='none',
                        line = dict(width = linew, color = cols),
                        opacity = opac
                )
    
    return trace_edges



def get_trace_edges_from_nodelist2D(G, l_nodes, posG, color, linew = 0.75, opac=0.1):
    '''
    Get trace of edges for plotting in 3D only for specific edges. 
    Input: 
    - G = Graph
    - posG = dictionary with nodes as keys and coordinates as values.
    - color = string; specific color to highlight specific edges; hex color
    
    Return a trace of specific edges. 
    '''
    l_spec_edges = [(u,v) for u,v in G.edges(l_nodes) if u in l_nodes and v in l_nodes]
   
    edge_x = []
    edge_y = []
    
    for edge in l_spec_edges:
            x0, y0 = posG[edge[0]]
            x1, y1 = posG[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
            

    trace_edges = pgo.Scatter(
                        x = edge_x, 
                        y = edge_y, 
                        mode = 'lines', hoverinfo='none',
                        line = dict(width = linew, color = color),
                        opacity = opac
                )
    
    return trace_edges



def plot_2D(data,path,fname):
    '''
    Create a 2D plot from traces using plotly.
    Input: 
    - data = list of traces
    - filename = string
    
    Return plot in 2D and file, saved as png.
    '''

    fig = pgo.Figure()
    
    for i in data:
        fig.add_trace(i)
        
    fig.update_layout(template= 'plotly_white', 
                      showlegend=False, width=1200, height=1200,
                          scene=dict(
                              xaxis_title='',
                              yaxis_title='',
                              xaxis=dict(nticks=0,tickfont=dict(
                                    color='white')),
                              yaxis=dict(nticks=0,tickfont=dict(
                                    color='white')),
                        ))    
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    
    # --- show figure ---
    #py.iplot(fig)
    
    # --- get html file ---  
    fig.write_html(path+fname+'.html')
    
    # --- get screenshot image (png) from html --- 
    #hti = Html2Image(output_path=path)
    #hti.screenshot(html_file = path+fname+'.html', save_as = fname+'.png')
    
    #not working with large file / time ! 
    #fig.write_image(fname+'.png') 
    
    return plotly.offline.plot(fig, filename = path+fname+'.html', auto_open=True)

        
########################################################################################

# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
#      ######     #######
#    ##     ##    ##    ##
#           ##    ##     ## 
#      #####      ##     ##
#           ##    ##     ##
#    ##     ##    ##    ##
#     ######      #######
#    
# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# E M B E D D I N G 
# -------------------------------------------------------------------------------------

def embed_tsne_3D(Matrix, prplxty, density, l_rate, n_iter, metric = 'cosine'):
    '''
    Dimensionality reduction from Matrix (t-SNE).
    Return dict (keys: node IDs, values: x,y,z).
    '''
    
    tsne3d = TSNE(n_components = 3, random_state = 0, perplexity = prplxty,
                     early_exaggeration = density,  learning_rate = l_rate, n_iter = n_iter, metric = metric,
                 square_distances=True)
    embed = tsne3d.fit_transform(Matrix)

    return embed 


def embed_umap_3D(Matrix, n_neighbors, spread, min_dist, metric='cosine', learn_rate = 1, n_ep = None):
    '''
    Dimensionality reduction from Matrix (UMAP).
    Return dict (keys: node IDs, values: x,y,z).
    '''

    n_components = 3 # for 3D

    U_3d = umap.UMAP(
        n_neighbors = n_neighbors,
        spread = spread,
        min_dist = min_dist,
        n_components = n_components,
        metric = metric,
        random_state=42,
        learning_rate = learn_rate, 
        n_epochs = n_ep)
    embed = U_3d.fit_transform(Matrix)
    
    return embed


def get_posG_3D(l_genes, embed):
    '''
    Generate coordinates from embedding. 
    Input:
    - l_genes = list of genes
    - embed = embedding from e.g. tSNE , UMAP ,... 
    
    Return dictionary with nodes as keys and coordinates as values in 3D. 
    '''
    
    posG = {}
    cc = 0
    for entz in l_genes:
        posG[entz] = (embed[cc,0],embed[cc,1],embed[cc,2])
        cc += 1
    
    return posG



def get_posG_3D_norm(G, DM, embed, r_scalingfactor=1.05):
    '''
    Generate coordinates from embedding. 
    Input:
    - G = Graph
    - DM = matrix 
    - embed = embedding from e.g. tSNE , UMAP ,... 
    
    Return dictionary with nodes as keys and coordinates as values in 3D normed. 
    '''
    genes = []
    for i in DM.index:
        if str(i) in G.nodes() or int(i) in G.nodes():
            genes.append(str(i))

    genes_rest = [] 
    for i in G.nodes():
        if i not in genes:
            genes_rest.append(str(i))
            
    posG_3Dumap = {}
    cc = 0
    for entz in genes:
        posG_3Dumap[entz] = (embed[cc,0],embed[cc,1],embed[cc,2])
        cc += 1

    #--------------------------------------------------------------
    # REST (if genes = G.nodes then rest will be ignored / empty)
    
    # center for sphere to arrange rest gene-datapoints
    xx=[]
    yy=[]
    zz=[]
    for i in posG_3Dumap.values():
        xx.append(i[0])
        yy.append(i[1])
        zz.append(i[2]) 

    cx = sum(xx)/len(genes)
    cy = sum(yy)/len(genes)
    cz = sum(zz)/len(genes)

    # generate spherical coordinates for rest genes (without e.g. GO term or Disease Annotation)
    indices = arange(0, len(genes_rest))
    phi = arccos(1 - 2*indices/len(genes_rest)) # 2* --> for both halfs of sphere (upper+lower)
    theta = pi * (1 + 5**0.5) * indices

    xm, ym, zm = max(posG_3Dumap.values())
    r = (math.sqrt((cx - xm)**2 + (cy - ym)**2 + (cz - zm)**2))*r_scalingfactor # +10 > ensure colored nodes within sphere
    x, y, z = cx+r*cos(theta) * sin(phi),cy+r*sin(theta) * sin(phi), cz+r*cos(phi)

    rest_points = []
    for i,j,k in zip(x,y,z):
        rest_points.append((i,j,k))

    posG_rest = dict(zip(genes_rest, rest_points))

    posG_all = {**posG_3Dumap, **posG_rest}
    posG_3D_complete_umap = {key:posG_all[key] for key in G.nodes()}

    # normalize coordinates 
    x_list3D = []
    y_list3D = []
    z_list3D = []
    for k,v in posG_3D_complete_umap.items():
        x_list3D.append(v[0])
        y_list3D.append(v[1])
        z_list3D.append(v[2])

    xx_norm3D = preprocessing.minmax_scale(x_list3D, feature_range=(0, 1), axis=0, copy=True)
    yy_norm3D = preprocessing.minmax_scale(y_list3D, feature_range=(0, 1), axis=0, copy=True)
    zz_norm3D = preprocessing.minmax_scale(z_list3D, feature_range=(0, 1), axis=0, copy=True)

    xx_norm3D_final=[]
    for i in xx_norm3D:
        xx_norm3D_final.append(round(i,10))

    yy_norm3D_final=[]
    for i in yy_norm3D:
        yy_norm3D_final.append(round(i,10))

    zz_norm3D_final=[]
    for i in zz_norm3D:
        zz_norm3D_final.append(round(i,10)) 

    posG_3D_complete_umap_norm = dict(zip(list(G.nodes()), zip(xx_norm3D_final,yy_norm3D_final,zz_norm3D_final)))
    
    return posG_3D_complete_umap_norm


def embed_umap_sphere(Matrix, n_neighbors, spread, min_dist):
    ''' 
    Generate spherical embedding of nodes in matrix input using UMAP.
    Input: 
    - Matrix = Feature Matrix with either all or specific  nodes (rows) and features (columns) or symmetric (nodes = rows and columns)
    - n_neighbors/spread/min_dist = floats; UMAP parameters.
    - metric = string; e.g. havervine, euclidean, cosine ,.. 
    
    Return sphere embedding. 
    '''
    
    model = umap.UMAP(
        n_neighbors = n_neighbors, 
        spread = spread,
        min_dist = min_dist,
        output_metric = 'haversine',
        random_state=42)
    sphere_mapper = model.fit(Matrix)

    return sphere_mapper



def get_posG_sphere_norm(G, DM, sphere_mapper, d_param, radius_rest_genes = 20):
    '''
    Generate coordinates from embedding. 
    Input:
    - G = Graph 
    - l_genes = list of node IDs, either specific or all nodes in the graph 
    - sphere_mapper = embedding from UMAP spherical embedding 
    - d_param = dictionary with nodes as keys and assigned radius as values 
    - radius_rest_genes = int; radius in case of genes e.g. not function associated if genes not all G.nodes()
    
    Return dictionary with nodes as keys and coordinates as values in 3D. 
    '''
    
    x = np.sin(sphere_mapper.embedding_[:, 0]) * np.cos(sphere_mapper.embedding_[:, 1])
    y = np.sin(sphere_mapper.embedding_[:, 0]) * np.sin(sphere_mapper.embedding_[:, 1])
    z = np.cos(sphere_mapper.embedding_[:, 0])
    
    genes = []
    for i in DM.index:
        if str(i) in G.nodes() or int(i) in G.nodes():
            genes.append(str(i))

    genes_rest = [] 
    for i in G.nodes():
        if i not in genes:
            genes_rest.append(i)
            
    posG_3Dsphere = {}
    cc = 0
    for entz in genes:
        posG_3Dsphere[entz] = (x[cc],y[cc], z[cc])
        cc += 1

    posG_3Dsphere_radius = {}
    for node,rad in d_param.items():
        for k,v in posG_3Dsphere.items():
            if k == node:
                posG_3Dsphere_radius[k] = (v[0]*rad, v[1]*rad, v[2]*rad)
 
    # generate spherical coordinates for rest genes (without e.g. GO term or Disease Annotation)
    indices = arange(0, len(genes_rest))
    phi = arccos(1 - 2*indices/len(genes_rest))
    theta = pi * (1 + 5**0.5) * indices

    r_rest = radius_rest_genes # radius for rest genes (e.g. if functional layout)
    x, y, z = r_rest*cos(theta) * sin(phi), r_rest*sin(theta) * sin(phi), r_rest*cos(phi)

    rest_points = []
    for i,j,k in zip(x,y,z):
        rest_points.append((i,j,k))

    posG_rest = dict(zip(genes_rest, rest_points))

    posG_all = {**posG_3Dsphere_radius, **posG_rest}
    posG_complete_sphere = {key:posG_all[key] for key in G.nodes()}

    # normalize coordinates 
    x_list = []
    y_list = []
    z_list = []
    for k,v in posG_complete_sphere.items():
        x_list.append(v[0])
        y_list.append(v[1])
        z_list.append(v[2])

    xx_norm = preprocessing.minmax_scale(x_list, feature_range=(0, 1), axis=0, copy=True)
    yy_norm = preprocessing.minmax_scale(y_list, feature_range=(0, 1), axis=0, copy=True)
    zz_norm = preprocessing.minmax_scale(z_list, feature_range=(0, 1), axis=0, copy=True)

    posG_complete_sphere_norm = dict(zip(list(G.nodes()), zip(xx_norm,yy_norm,zz_norm)))
    
    return posG_complete_sphere_norm



# -------------------------------------------------------------------------------------
# P L O T T I N G 
# -------------------------------------------------------------------------------------


def get_trace_nodes_3D(posG, info_list, color, size, opac=0.9):
    '''
    Get trace of nodes for plotting in 3D. 
    Input: 
    - posG = dictionary with nodes as keys and coordinates as values.
    - info_list = hover information for each node, e.g. a list sorted according to the initial graph/posG keys
    - color = string; hex color
    - opac = transparency of edges e.g. 0.2
    
    Return a trace for plotly graph objects plot. 
    '''
    
    key_list=list(posG.keys())
    trace = pgo.Scatter3d(x=[posG[key_list[i]][0] for i in range(len(key_list))],
                           y=[posG[key_list[i]][1] for i in range(len(key_list))],
                           z=[posG[key_list[i]][2] for i in range(len(key_list))],
                           mode = 'markers',
                           text = info_list,
                           hoverinfo = 'text',
                           #textposition='middle center',
                           marker = dict(
                color = color,
                size = size,
                symbol = 'circle',
                line = dict(width = 1.0,
                        color = color),
                opacity = opac,
            ),
        )
    
    return trace


def get_trace_edges_3D(G, posG, color, opac = 0.2, linewidth=0.2):
    '''
    Get trace of edges for plotting in 3D. 
    Input: 
    - G = Graph
    - posG = dictionary with nodes as keys and coordinates as values.
    - color = string; hex color
    - opac = transparency of edges e.g. 0.2
    
    Return a trace for plotly graph objects plot. 
    '''
    
    edge_x = []
    edge_y = []
    edge_z = []
    for edge in G.edges():
            x0, y0, z0 = posG[edge[0]]
            x1, y1, z1 = posG[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
            edge_z.append(z0)
            edge_z.append(z1)
            edge_z.append(None)

    trace_edges = pgo.Scatter3d(
                                x = edge_x, 
                                y = edge_y, 
                                z = edge_z,
                                mode = 'lines', hoverinfo='none',
                                line = dict(width = linewidth, color = color),
                                opacity = opac
                        )

    return trace_edges



def get_trace_edges_from_nodelist3D(G, l_genes, posG, color, linew = 0.75, opac=0.1):
    '''
    Get trace of edges for plotting in 3D only for specific edges. 
    Input: 
    - G = Graph
    - posG = dictionary with nodes as keys and coordinates as values.
    - color = string; specific color to highlight specific edges; hex color
    
    Return a trace of specific edges. 
    '''
    l_spec_edges = [(u,v)for u,v in G.edges(l_genes) if u in l_genes and v in l_genes]
    
    edge_x = []
    edge_y = []
    edge_z = []
    for edge in l_spec_edges:
            x0, y0,z0 = posG[edge[0]]
            x1, y1,z1 = posG[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
            edge_z.append(z0)
            edge_z.append(z1)
            edge_z.append(None)
            
    trace_edges = pgo.Scatter3d(
                        x = edge_x, 
                        y = edge_y, 
                        z = edge_z,
                        mode = 'lines', hoverinfo='none',
                        line = dict(width = linew, color = color),
                        opacity = opac
                )
    return trace_edges


def get_trace_edges_specific3D(d_edges_col, posG, linew = 0.75, opac=0.1):

    edge_x = []
    edge_y = []
    edge_z = []
    for edge, col in d_edges_col.items():
            x0, y0,z0 = posG[edge[0]]
            x1, y1,z1 = posG[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
            edge_z.append(z0)
            edge_z.append(z1)
            edge_z.append(None)
            
    color = list(d_edges_col.values())[0]
    
    trace_edges = pgo.Scatter3d(
                        x = edge_x, 
                        y = edge_y, 
                        z = edge_z,
                        mode = 'lines', hoverinfo='none',
                        line = dict(width = linew, color = color),
                        opacity = opac
                )
    
    return trace_edges


def get_trace_edges_landscape(x,y,z0,z):
    '''
    Create trace of vertical connecting edges in between node z0 and node z=parameter (e.g.disease count).
    Return trace with edges.
    '''
    
    Xe = []
    for u in x:
        Xe += [u,u,None]

    Ye = []   
    for v in y:
        Ye += [v,v,None]  

    Ze = []  
    for w in z0:
        for t in z:
            Ze += [w,t,None]
            
    trace_edge = pgo.Scatter3d(
        x = Xe, 
        y = Ye, 
        z = Ze,
        mode = 'lines', hoverinfo='none',
        line = dict(width = 3.0, color = 'darkgrey'),
        opacity = 0.5
    )

    return trace_edge


def plot_3D(data,path,fname, scheme='light',annotat=None):
    '''
    Create a 3D plot from traces using plotly.
    Input: 
    - data = list of traces
    - filename = string
    - scheme = 'light' or 'dark'
    - annotations = None or plotly annotations
    
    Return plot in 3D and file, saved as html.
    '''

    fig = pgo.Figure()
    
    for i in data:
        fig.add_trace(i)
      
    if scheme == 'dark' and annotat==None:
        fig.update_layout(template='plotly_dark', showlegend=False, autosize = True,
                          scene=dict(
                              xaxis_title='',
                              yaxis_title='',
                              zaxis_title='',
                              xaxis=dict(nticks=0,tickfont=dict(
                                    color='black')),
                              yaxis=dict(nticks=0,tickfont=dict(
                                    color='black')),
                              zaxis=dict(nticks=0,tickfont=dict(
                                    color='black')),
                            dragmode="turntable"
                        )) 
        
    elif scheme == 'dark':    
        fig.update_layout(template='plotly_dark', showlegend=False, autosize = True,
                                  scene=dict(
                                      xaxis_title='',
                                      yaxis_title='',
                                      zaxis_title='',
                                      xaxis=dict(nticks=0,tickfont=dict(
                                            color='black')),
                                      yaxis=dict(nticks=0,tickfont=dict(
                                            color='black')),
                                      zaxis=dict(nticks=0,tickfont=dict(
                                            color='black')),
                                    dragmode="turntable",
                                    annotations=annotat,
                                ))

    elif scheme == 'light' and annotat==None:
        fig.update_layout(template='plotly_white', showlegend=False, width=1200, height=1200,
                          scene=dict(
                              xaxis_title='',
                              yaxis_title='',
                              zaxis_title='',
                              xaxis=dict(nticks=0,tickfont=dict(
                                    color='white')),
                              yaxis=dict(nticks=0,tickfont=dict(
                                    color='white')),
                              zaxis=dict(nticks=0,tickfont=dict(
                                    color='white')),    
                            dragmode="turntable",
                        ))    
        
    elif scheme == 'light':
        fig.update_layout(template='plotly_white', showlegend=False, width=1200, height=1200,
                          scene=dict(
                              xaxis_title='',
                              yaxis_title='',
                              zaxis_title='',
                              xaxis=dict(nticks=0,tickfont=dict(
                                    color='white')),
                              yaxis=dict(nticks=0,tickfont=dict(
                                    color='white')),
                              zaxis=dict(nticks=0,tickfont=dict(
                                    color='white')),    
                            dragmode="turntable",
                            annotations = annotat
                        ))    

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    
    # --- show figure ---
    #py.iplot(fig)
    
    # --- get html file ---  
    #fig.write_html(path+fname+'.html')
    
    return plotly.offline.plot(fig, filename = path+fname+'.html', auto_open=True)



def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)


def color_nodes(l_genes, color):
    ''' 
    Color nodes of list with same color.
    Returns as dict with node ID and assigned color. 
    ''' 
    d_col = {}
    for node in l_genes:
        d_col[str(node)] = color
    
    return d_col


def generate_colorlist_nodes(n):
    '''
    Generate color list based on color count (i.e. nodes to be colored).
    Input:
    - n = number of colors to generate.
    
    Return list of colors.
    '''
    colors = [colorsys.hsv_to_rgb(1.0/n*x,1,1) for x in range(n)]
    color_list = []
    for c in colors:
        cc = [int(y*255) for y in c]
        color_list.append('#%02x%02x%02x' % (cc[0],cc[1],cc[2]))
        
    return color_list


def hex_to_rgb(hx):
    hx = hx.lstrip('#')
    hlen = len(hx)
    return tuple(int(hx[i:i+hlen//3], 16) for i in range(0, hlen, hlen//3))


def adjust_color_lightness(r, g, b, factor):
    h, l, s = rgb2hls(r / 255.0, g / 255.0, b / 255.0)
    l = max(min(l * factor, 1.0), 0.0)
    r, g, b = hls2rgb(h, l, s)
    return rgb2hex(int(r * 255), int(g * 255), int(b * 255))


def darken_color(r, g, b, factor=0.9):
    return adjust_color_lightness(r, g, b, 1 - factor)


def color_nodes_from_dict_unsort(d_to_be_colored, palette):
    ''' 
    Generate node colors based on dictionary.
    Input: 
    - d_to_be_colored = dictionary with nodes as keys and values indicating different colors.
    - palette = sns.color palette e.g. 'YlOrRd' 
    
    Return dictionary (randomly sorted) with nodes as keys and assigned color to each node.
    ''' 

    # coloringg
    color_groups = set(d_to_be_colored.values())
    color_count = len(color_groups)
    pal = sns.color_palette(palette, color_count)
    palette = pal.as_hex()

    d_colorgroups = {}
    for n in color_groups:
        d_colorgroups[n] = [k for k in d_to_be_colored.keys() if d_to_be_colored[k] == n]
        
    d_colorgroups_sorted = {key:d_colorgroups[key] for key in sorted(d_colorgroups.keys())}

    d_val_col = {}
    for idx,val in enumerate(d_colorgroups_sorted):
        for ix,v in enumerate(palette):
            if idx == ix:
                d_val_col[val] = v

    d_node_color = {}
    for y in d_to_be_colored.items(): # y[0] = node id, y[1] = val
        for x in d_val_col.items(): # x[0] = val, x[1] = (col,col,col)
            if x[0] == y[1]:
                d_node_color[y[0]]=x[1]
    
    return d_node_color # colors


def color_nodes_from_dict(G, d_to_be_colored, palette): 
    ''' 
    Generate node colors based on dictionary.
    Input: 
    - G = Graph 
    - d_to_be_colored = dictionary with nodes as keys and values indicating different colors.
    - palette = sns.color palette e.g. 'YlOrRd' 
    
    Return dictionary, sorted according to Graph nodes, with nodes as keys and assigned color to each node.
    ''' 
    
    # coloring
    color_groups = set(d_to_be_colored.values())
    color_count = len(color_groups)
    pal = sns.color_palette(palette, color_count)
    palette = pal.as_hex()

    d_colorgroups = {}
    for n in color_groups:
        d_colorgroups[n] = [k for k in d_to_be_colored.keys() if d_to_be_colored[k] == n]
        
    d_colorgroups_sorted = {key:d_colorgroups[key] for key in sorted(d_colorgroups.keys())}

    d_val_col = {}
    for idx,val in enumerate(d_colorgroups_sorted):
        for ix,v in enumerate(palette):
            if idx == ix:
                d_val_col[val] = v

    d_node_color = {}
    for y in d_to_be_colored.items(): # y[0] = node id, y[1] = val
        for x in d_val_col.items(): # x[0] = val, x[1] = (col,col,col)
            if x[0] == y[1]:
                d_node_color[y[0]]=x[1]

    # SORT dict based on G.nodes
    d_node_color_sorted = dict([(key, d_node_color[key]) for key in G.nodes()])
    
    return d_node_color_sorted


def color_nodes_from_list(G, l_nodes, col):
    '''
    Color nodes based on essentiality state.
    Input: 
    - G = graph
    - l_nodes = list of nodes
    - col = string or hex; color 
    All rest genes will be colored in grey.
    
    Return list of colors for each node in the graph, sorted based on Graph nodes.
    '''

    d_nodes = {}
    for node in l_nodes:
        d_nodes[node] = col

    d_restnodes = {}
    for i in G.nodes():
        if i not in d_nodes.keys():
            d_restnodes[i] = 'lightgrey'

    d_all_nodes = {**d_nodes, **d_restnodes}
    d_all_nodes_sorted = {key:d_all_nodes[key] for key in G.nodes()}   
    
    return d_all_nodes_sorted



def color_edges_from_nodelist(G, l_nodes, color_main, color_rest): # former: def color_disease_outgoingedges(G, l_majorcolor_nodes, color)
    '''
    Color (highlight) edges from specific node list.
    Input: 
    - G = Graph 
    - l_nodes = list of nodes 
    - color = color to hightlight
    All other edges will remain in grey.
    
    Return edge list sorted based on G.edges() 
    '''
    
    edge_lst = []
    for edge in G.edges():
        for e in edge:
            if e in l_nodes:
                edge_lst.append(edge)

    d_col_edges = {}
    for e in edge_lst:
        d_col_edges[e]=color_main
    
    d_grey_edges = {}
    for edge in G.edges():
        if edge not in d_col_edges.keys(): 
            d_grey_edges[edge] = color_rest # '#d3d3d3'

    d_edges_all = {**d_col_edges, **d_grey_edges}
    d_edges_all_sorted = {key:d_edges_all[key] for key in G.edges()}

    edge_color = list(d_edges_all_sorted.values())
    
    return  d_edges_all


def color_edges_from_node(G, node, color):
    '''
    Color (highlight) edges from specific node list exclusively.
    Input:
    - G = Graph 
    - l_nodes = list of nodes 
    - color = string; color to hightlight
    
    Return edge list for selected edges IF ONE node is IN l_genes. 
    '''
    
    edge_lst = [(u,v)for u,v in G.edges(node) if u in node or v in node]

    d_col_edges = {}
    for e in edge_lst:
        d_col_edges[e]=color
    return d_col_edges


def color_edges_from_nodelist_specific(G, l_nodes, color):
    '''
    Color (highlight) edges from specific node list exclusively.
    Input:
    - G = Graph 
    - l_nodes = list of nodes 
    - color = string; color to hightlight
    
    Return edge list for selected edges IF ONE node is IN l_genes. 
    '''
    
    edge_lst = [(u,v)for u,v in G.edges(l_nodes) if u in l_nodes and v in l_nodes]

    d_col_edges = {}
    for e in edge_lst:
        d_col_edges[e]=color
    return d_col_edges


def color_edges_from_nodelist_specific_or(G, l_nodes, color):
    '''
    Color (highlight) edges from specific node list exclusively.
    Input:
    - G = Graph 
    - l_nodes = list of nodes 
    - color = string; color to hightlight
    
    Return edge list for selected edges IF ONE node is IN l_genes. 
    '''
    
    edge_lst = [(u,v)for u,v in G.edges(l_nodes) if u in l_nodes or v in l_nodes]

    d_col_edges = {}
    for e in edge_lst:
        d_col_edges[e]=color
    return d_col_edges


def colors_spectralclustering(G, posG, n_clus, n_comp, pal ='gist_rainbow'):
    '''
    Generate node colors based on clustering.
    Input:
    - G = Graph
    - posG = dictionary with nodes as keys and xy(z) coordinates
    - n_clus = int; number of clusters
    - n_comp = int; number of components (e.g. 10)
    - palette(optional) = string; sns color palette e.g. "gist_rainbow"

    Returns a dictionary with nodes as keys and color values based on clustering method. 
    '''
    
    df_posG = pd.DataFrame(posG).T 

    model = SpectralClustering(n_clusters=n_clus,n_components = n_comp, affinity='nearest_neighbors',random_state=0)
    clusterid = model.fit(df_posG)
    d_node_clusterid = dict(zip(genes, clusterid.labels_))

    colors_unsort = color_nodes_from_dict_unsort(d_node_clusterid, pal) #'ocean'
    genes_val = ['#696969']*len(genes_rest)
    colors_rest = dict(zip(genes_rest, genes_val))
    colors_all = {**colors_rest, **colors_unsort}

    d_colors = {key:colors_all[key] for key in G.nodes}
    
    return d_colors



def colors_dbscanclustering(G, DM, posG, epsi, min_sam, pal = 'gist_rainbow', col_rest = '#696969'):
    '''
    Generate node colors based on clustering.
    Input:
    - G = Graph
    - posG = dictionary with nodes as keys and xy(z) coordinates
    - epsi = int; number of clusters
    - min_sam = int; number of components (e.g. 10)
    - palette(optional) = string; sns color palette e.g. "gist_rainbow"

    Returns a dictionary with nodes as keys and color values based on clustering method. 
    '''
    genes = []
    for i in DM.index:
        if str(i) in G.nodes():
            genes.append(str(i))

    genes_rest = [] 
    for g in G.nodes():
        if str(g) not in genes:
            genes_rest.append(g)
            
    df_posG = pd.DataFrame(posG).T 
    dbscan = DBSCAN(eps=epsi, min_samples=min_sam) 
    clusterid = dbscan.fit(df_posG)
    d_node_clusterid = dict(zip(genes, clusterid.labels_))

    colors_unsort = color_nodes_from_dict_unsort(d_node_clusterid, pal)
    genes_val = [col_rest]*len(genes_rest)
    colors_rest = dict(zip(genes_rest, genes_val))
    colors_all = {**colors_rest, **colors_unsort}

    d_colors_sorted = {key:colors_all[key] for key in G.nodes}
    print('Number of Clusters: ', len(set(clusterid.labels_)))
    
    return d_colors_sorted



def kmeansclustering(posG, n_clus):
    
    df_posG = pd.DataFrame(posG).T 
    kmeans = KMeans(n_clusters=n_clus, random_state=0).fit(df_posG)
    centrs = kmeans.cluster_centers_
    
    return kmeans, centrs



def colors_kmeansclustering(G, DM, kmeans, pal = 'gist_rainbow'):
    '''
    Generate node colors based on clustering.
    Input:
    - G = Graph
    - posG = dictionary with nodes as keys and xy(z) coordinates
    - n_clus = int; number of clusters
    - palette(optional) = string; sns color palette e.g. "gist_rainbow"

    Returns a dictionary with nodes as keys and color values based on clustering method. 
    '''

    genes = []
    for i in DM.index:
        if str(i) in G.nodes():
            genes.append(str(i))

    genes_rest = [] 
    for g in G.nodes():
        if str(g) not in genes:
            genes_rest.append(g)
            
    d_node_clusterid = dict(zip(genes, kmeans.labels_))
    colors_unsort = color_nodes_from_dict_unsort(d_node_clusterid, pal ) #'prism'
    
    genes_val = ['#696969']*len(genes_rest)
    colors_rest = dict(zip(genes_rest, genes_val))
    colors_all = {**colors_rest, **colors_unsort}
    d_colors_sorted = {key:colors_all[key] for key in G.nodes}
    
    return d_colors_sorted



# -------------------------------------------------------------------------------------
# E S S E N T I A L I T Y   S P E C I F I C  
# -------------------------------------------------------------------------------------


def color_essentiality_nodes(G, essentials, nonessentials, color1, color2):
    '''
    Color nodes based on essentiality state.
    Input: 
    - G = graph
    - essentials = list of all essential genes
    - nonessentials = list of all non-essential genes 
    - color1 = string; to color essential genes
    - color2 = string; to color non-essential genes 
    All rest genes will be colored in grey.
    
    Return list of colors for each node in the graph, sorted based on Graph nodes.
    '''

    d_ess = {}
    for node in essentials:
        d_ess[node] = color1

    d_no_ess = {}
    for node in nonessentials:
        d_no_ess[node] = color2

    d_essentiality = {**d_ess, **d_no_ess}

    d_restnodes = {}
    for i in G.nodes():
        if i not in d_essentiality.keys():
            d_restnodes[i] = 'grey'

    d_all_essentiality = {**d_essentiality, **d_restnodes}
    d_all_nodes_sorted = {key:d_all_essentiality[key] for key in G.nodes()}   
    
    return d_all_nodes_sorted


def zparam_essentiality(G, essential_genes, non_ess_genes, value_ess, value_noness, value_undef):
    '''
    Generate z-heights for each node based on essentiality. 
    Input: 
    - G = graph
    - essential_genes = list of all essential genes
    - non_ess_genes = list of all non-essential genes 
    - value_ess = integer; z-height parameter
    - value_noness = integer; z-height parameter
    - value_undef = integer; z-height parameter
    
    Return dictionary with nodes as keys and z-heights assigned according to essentiality state. 
    '''
    
    d_ess = {}
    for node in essential_genes:
        d_ess[node] = value_ess

    d_no_ess = {}
    for node in non_ess_genes:
        d_no_ess[node] = value_noness

    d_essentiality = {**d_ess, **d_no_ess}

    d_restnodes = {}
    for i in G.nodes():
        if i not in d_essentiality.keys():
            d_restnodes[i] = value_undef
            
    d_all_essentiality = {**d_essentiality, **d_restnodes}
    d_all_nodes_sorted = {key:d_all_essentiality[key] for key in G.nodes()}   
    
    return d_all_nodes_sorted


# -------------------------------------------------------------------------------------
# D I S E A S E   S P E C I F I C
# -------------------------------------------------------------------------------------


def get_disease_genes(G, d_names_do, d_do_genes, disease_category):
    ''' 
    Get disease-specific genes. 
    Input: 
    - G = Graph 
    - d_names_do: dictionary with gene symbol as keys and disease annotations as values 
    - d_do_genes: dictionary with disease as key and list of genes associated with disease as values
    - disease_category: string; specify disease category e.g. 'cancer'
    
    Return a list of genes associated to specified disease as set for no duplicates.
    '''
    
    # get all genes from disease category
    l_disease_genes = [] 
    for d_name in d_names_do.keys():
        if d_name.find(disease_category) != -1:
            try:
                l_genes = d_do_genes[d_names_do[d_name]]
                for gene in l_genes:
                    l_disease_genes.append(gene)
            except:
                    pass
                
    set_disease_genes = set(l_disease_genes)
    
    return set_disease_genes


# -------------------------------------------------------------------------------------
# H U B   S P E C I F I C
# -------------------------------------------------------------------------------------

   
def get_hubs(G, max_treshold, min_treshold):
    
    d_degree = dict(nx.degree(G))

    hubs = {}
    for k,v in d_degree.items():
        if v >= min_treshold and v <= max_treshold:
            hubs[k] = v
    #print('Hubs: ',hubs)

    # get their neighbours
    neighbours = {}
    hubs_neigh = []
    for i in hubs.keys():
        for edge in G.edges():
            if edge[0] == i:
                hubs_neigh.append(edge[1])
            elif edge[1] == i:
                hubs_neigh.append(edge[0])
        neighbours[i] = hubs_neigh
    print('num of neighbors:', len(neighbours))
    
    return hubs,neighbours



def color_nodes_hubs(G, hubs, neighs, hubs_col_nodes, neigh_col_nodes):
    
    rest_col_nodes = '#d3d3d3' 

    colors_hubs = {}
    for i in G.nodes():
        if str(i) in hubs.keys():
            colors_hubs[i] = hubs_col_nodes
        elif str(i) in neighs.keys():
            colors_hubs[i] = neigh_col_nodes
        else: 
            colors_hubs[i] = rest_col_nodes

    hubs_all_sorted = {key:colors_hubs[key] for key in G.nodes()}
    #colors = list(hubs_all_sorted.values())
    
    return hubs_all_sorted 



########################################################################################
#
# N O D E  S I Z E   F U N C T I O N S
#
########################################################################################


# -------------------------------------------------------------------------------------
# DEGREE SPECIFIC
# -------------------------------------------------------------------------------------

def draw_node_degree(G, scalef):
    '''
    Calculate the node degree from graph positions (dict).
    Return list of radii for each node (2D). 
    '''
    
    #x = 20
    #ring_frac = np.sqrt((x-1.)/x)
    #ring_frac = (x-1.)/x

    l_size = {}
    for node in G.nodes():
        k = nx.degree(G, node)
        R = scalef * (1 + k**1.1) 

        l_size[node] = R
        
    return l_size


def draw_node_degree_3D(G, scalef):
    '''
    Calculate the node degree from graph positions (dict).
    Return list of sizes for each node (3D). 
    '''
    
    x = 3
    ring_frac = (x-1.)/x

    deg = dict(G.degree())
    
    d_size = {}
    for i in G.nodes():
        for k,v in deg.items():
            if i == k:
                R = scalef * (1+v**0.9)
                r = ring_frac * R
                d_size[i] = R
    
    return d_size 




















########################################################################################
#
# E M B E D D I N G & P L O T T I N G  2D + 3D
#
########################################################################################


# -------------------------------------------------------------------------------------
#
#      ######     #######
#    ##     ##    ##    ##
#           ##    ##     ##
#          ##     ##     ##
#        ##       ##     ##
#      ##         ##     ##
#    ##           ##    ##
#    ##########   #######
#
# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# E M B E D D I N G
# -------------------------------------------------------------------------------------


def embed_tsne_2D(Matrix, prplxty, density, l_rate, steps, metric = 'cosine'):
    '''
    Dimensionality reduction from Matrix using t-SNE.
    Return dict (keys: node IDs, values: x,y).
    '''

    tsne = TSNE(n_components = 2, random_state = 0, perplexity = prplxty, metric = metric, init='pca',
                     early_exaggeration = density,  learning_rate = l_rate ,n_iter = steps, square_distances=True)

    embed = tsne.fit_transform(Matrix)

    return embed


def embed_umap_2D(Matrix, n_neigh, spre, m_dist, metric='cosine', learn_rate = 1, n_ep = None):
    '''
    Dimensionality reduction from Matrix using UMAP.
    Return dict (keys: node IDs, values: x,y).
    '''
    n_comp = 2

    U = umap.UMAP(
        n_neighbors = n_neigh,
        spread = spre,
        min_dist = m_dist,
        n_components = n_comp,
        metric = metric,
        random_state=42,
        learning_rate = learn_rate,
        n_epochs = n_ep)

    embed = U.fit_transform(Matrix)

    return embed


def get_posG_2D(l_nodes, embed):
    '''
    Get 2D coordinates for each node.
    Return dict with node: x,y coordinates.
    '''

    posG = {}
    cc = 0
    for entz in l_nodes:
        # posG[str(entz)] = (embed[cc,0],embed[cc,1])
        posG[entz] = (embed[cc,0],embed[cc,1])
        cc += 1

    return posG


def get_posG_2D_norm_old(G, DM, embed, r_scalingfactor = 5):
    '''
    Generate coordinates from embedding.
    Input:
    - G = Graph
    - DM = matrix
    - embed = embedding from e.g. tSNE , UMAP ,...

    Return dictionary with nodes as keys and coordinates as values in 3D normed.
    '''

    genes = []
    for i in DM.index:
        if str(i) in G.nodes():
            genes.append(str(i))

    genes_rest = []
    for g in G.nodes():
        if str(g) not in genes:
            genes_rest.append(g)

    posG = {}
    cc = 0
    for entz in genes:
        posG[entz] = (embed[cc,0],embed[cc,1])
        cc += 1

    #--------------------------------------------------------------
    # REST (if genes = G.nodes then rest will be ignored / empty)

    # generate circle coordinates for rest genes (without e.g. GO term or Disease Annotation)
    t = np.random.uniform(0,2*np.pi,len(genes_rest))

    xx=[]
    yy=[]
    for i in posG.values():
        xx.append(i[0])
        yy.append(i[1])

    cx = np.mean(xx)
    cy = np.mean(yy)

    xm, ym = max(posG.values())
    r = (math.sqrt((xm-cx)**2 + (ym-cy)**2))*r_scalingfactor #*1.05 # multiplying with 1.05 makes cirle larger to avoid "outsider nodes/genes"

    x = r*np.cos(t)
    y = r*np.sin(t)
    rest = []
    for i,j in zip(x,y):
            rest.append((i,j))

    posG_rest = dict(zip(genes_rest, rest))

    posG_all = {**posG, **posG_rest}
    posG_complete = {key:posG_all[key] for key in G.nodes()}

    # normalize coordinates
    x_list = []
    y_list = []
    for k,v in posG_complete.items():
        x_list.append(v[0])
        y_list.append(v[1])

    xx_norm = sklearn.preprocessing.minmax_scale(x_list, feature_range=(0, 1), axis=0, copy=True)
    yy_norm = sklearn.preprocessing.minmax_scale(y_list, feature_range=(0, 1), axis=0, copy=True)

    xx_norm_final=[]
    for i in xx_norm:
        xx_norm_final.append(round(i,10))

    yy_norm_final=[]
    for i in yy_norm:
        yy_norm_final.append(round(i,10))

    posG_complete_norm = dict(zip(list(G.nodes()),zip(xx_norm_final,yy_norm_final)))

    return posG_complete_norm



def labels2D(posG, feature_dict):
    '''
    Create Node Labels, based on a dict of coordinates (keys:node ID, values: x,y)
    Return new dict of node iDs and features for each node.
    '''

    labels = {}
    c = 0
    for node, xy in sorted(posG.items(), key = lambda x: x[1][0]):
        labels[node] = ([node,feature_dict[node][0],feature_dict[node][1],feature_dict[node][2],feature_dict[node][3]])
        c+=1

    return labels


def position_labels(posG, move_x, move_y):
    '''
    Create label position of coordinates dict.
    Return new dict with label positions.
    '''

    posG_labels = {}
    for key,val in posG.items():
        xx = val[0] + move_x
        yy = val[1] + move_y
        posG_labels[key] = (xx,yy)

    return posG_labels


# -------------------------------------------------------------------------------------
# P L O T T I N G
# -------------------------------------------------------------------------------------

def get_trace2D(x,y,trace_name,colour):
    '''
    Get trace 2D.
    Used for distance functions (2D)
    '''

    trace = pgo.Scatter(name = trace_name,
    x = x,
    y = y,
    mode='markers',
    marker=dict(
        size=6,
        color=colour
    ),)

    return trace


def get_trace_nodes_2D(posG, info_list, color_list, size, opac=0.9):
    '''
    Get trace of nodes for plotting in 2D.
    Input:
    - G = Graph
    - posG = dictionary with nodes as keys and coordinates as values.
    - color_list = list of colours obtained from any color function (see above sections).
    - opacity = transparency of edges e.g. 0.2

    Return a trace for plotly graph objects plot.
    '''

    key_list=list(posG.keys())
    trace = pgo.Scatter(x=[posG[key_list[i]][0] for i in range(len(key_list))],
                           y=[posG[key_list[i]][1] for i in range(len(key_list))],
                           mode = 'markers',
                           text = info_list,
                           hoverinfo = 'text',
                           #textposition='middle center',
                           marker = dict(
                color = color_list,
                size = size,
                opacity = opac,
                symbol = 'circle',
                line = dict(width = 0.4,
                        color = 'dimgrey',
                        )
            ),
        )

    return trace



def get_trace_edges_from_nodelist2D_old(l_spec_edges, posG, col, opac=0.1):
    '''
    Get trace of edges for plotting in 3D only for specific edges.
    Input:
    - G = Graph
    - posG = dictionary with nodes as keys and coordinates as values.
    - color = string; specific color to highlight specific edges

    Return a trace of specific edges.
    '''

    edge_x = []
    edge_y = []
    for edge in l_spec_edges:
            x0, y0 = posG[edge[0]]
            x1, y1 = posG[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

    trace_edges = pgo.Scatter(
                        x = edge_x,
                        y = edge_y,
                        mode = 'lines', hoverinfo='none',
                        line = dict(width = 0.1, color = [col]*len(edge_x)),
                        opacity = opac
                )

    return trace_edges


def get_trace_edges_from_nodelist2D(l_spec_edges, posG, col, opac=0.1):
    '''
    Get trace of edges for plotting in 3D only for specific edges.
    Input:
    - G = Graph
    - posG = dictionary with nodes as keys and coordinates as values.
    - color = string; specific color to highlight specific edges

    Return a trace of specific edges.
    '''

    edge_x = []
    edge_y = []
    for edge in l_spec_edges:
            x0, y0 = posG[edge[0]]
            x1, y1 = posG[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

    trace_edges = pgo.Scatter(
                        x = edge_x,
                        y = edge_y,
                        mode = 'lines', hoverinfo='none',
                        line = dict(width = 0.1, color = col),#[col]*len(edge_x)),
                        opacity = opac
                )

    return trace_edges



def plot_2D(data,path,fname):
    '''
    Create a 3D plot from traces using plotly.
    Input:
    - data = list of traces
    - filename = string

    Return plot in 2D and file, saved as png.
    '''

    fig = pgo.Figure()

    for i in data:
        fig.add_trace(i)

    fig.update_layout(template= 'plotly_white',
                      showlegend=False, width=1200, height=1200,
                          scene=dict(
                              xaxis_title='',
                              yaxis_title='',
                              xaxis=dict(nticks=0,tickfont=dict(
                                    color='white')),
                              yaxis=dict(nticks=0,tickfont=dict(
                                    color='white')),
                        ))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    # --- show figure ---
    #py.iplot(fig)

    # --- get html file ---
    fig.write_html(path+fname+'.html')

    # --- get screenshot image (png) from html ---
    hti = Html2Image(output_path=path)
    hti.screenshot(html_file = path+fname+'.html', save_as = fname+'.png')

    #not working with large file / time !
    #fig.write_image(fname+'.png')

    return #py.iplot(fig)




# -------------------------------------------------------------------------------------
#
#      ######     #######
#    ##     ##    ##    ##
#           ##    ##     ##
#      #####      ##     ##
#           ##    ##     ##
#    ##     ##    ##    ##
#     ######      #######
#
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# E M B E D D I N G
# -------------------------------------------------------------------------------------

def embed_tsne_3D(Matrix, prplxty, density, l_rate, n_iter, metric = 'cosine'):
    '''
    Dimensionality reduction from Matrix (t-SNE).
    Return dict (keys: node IDs, values: x,y,z).
    '''

    tsne3d = TSNE(n_components = 3, random_state = 0, perplexity = prplxty,
                     early_exaggeration = density,  learning_rate = l_rate, n_iter = n_iter, metric = metric,
                     square_distances=True)
    embed = tsne3d.fit_transform(Matrix)

    return embed


def embed_tsne_3D_test(Matrix, prplxty, density, l_rate, n_iter, metric = 'cosine'):
    '''
    Dimensionality reduction from Matrix (t-SNE).
    Return dict (keys: node IDs, values: x,y,z).
    '''
    tsne3d = TSNE(n_components = 3, random_state = 0, perplexity = prplxty,
                     early_exaggeration = density,  learning_rate = l_rate, n_iter = n_iter, metric = metric,
                     square_distances=True)
    embed = tsne3d.fit(Matrix)

    return embed.embedding_


def embed_umap_3D(Matrix, n_neighbors, spread, min_dist, metric='cosine', learn_rate = 1, n_ep = None):
    '''
    Dimensionality reduction from Matrix (UMAP).
    Return dict (keys: node IDs, values: x,y,z).
    '''
    #print('CSDEBUG: got to embed_umap_3D')
    n_components = 3 # for 3D

    U_3d = umap.UMAP(
        n_neighbors = n_neighbors,
        spread = spread,
        min_dist = min_dist,
        n_components = n_components,
        metric = metric,
        random_state=42,
        learning_rate = learn_rate,
        n_epochs = n_ep)
    #print('CSDEBUG: UMAP complete in embed_umap_3D')
    #embed = U_3d.fit_transform(Matrix)
    embed = U_3d.fit(Matrix)
    #print('CSDEBUG: fit complete in embed_umap_3D')
    embed = U_3d.transform(Matrix)
    #print('CSDEBUG: transform complete in embed_umap_3D')
    #print('CSDEBUG: fit_transform complete in embed_umap_3D')

    return embed



def embed_umap_3D_test(Matrix, n_neighbors, spread, min_dist, metric='cosine', learn_rate = 1, n_ep = None):
    '''
    Dimensionality reduction from Matrix (UMAP).
    Return dict (keys: node IDs, values: x,y,z).
    '''
    print('CSDEBUG: got to embed_umap_3D')
    n_components = 3 # for 3D

    U_3d = umap.UMAP(
        n_neighbors = n_neighbors,
        spread = spread,
        min_dist = min_dist,
        n_components = n_components,
        metric = metric,
        random_state=42,
        learning_rate = learn_rate,
        n_epochs = n_ep)

    print('CSDEBUG: UMAP complete in embed_umap_3D')
    pre_embed = U_3d.fit(Matrix)
    #print('CSDEBUG: fit complete in embed_umap_3D')
    embed = pre_embed.transform(Matrix)
    #print('CSDEBUG: transform complete in embed_umap_3D')
    #print('CSDEBUG: fit_transform complete in embed_umap_3D')

    return embed



def get_posG_3D_norm_old(G, DM, embed):
    '''
    Generate coordinates from embedding.
    Input:
    - G = Graph
    - DM = matrix
    - embed = embedding from e.g. tSNE , UMAP ,...

    Return dictionary with nodes as keys and coordinates as values in 3D normed.
    '''

    genes = []
    for i in DM.index:
        if str(i) in G.nodes():
            genes.append(str(i))

    genes_rest = []
    for g in G.nodes():
        if str(g) not in genes:
            genes_rest.append(g)


    posG_3Dumap = {}
    cc = 0
    for entz in genes:
        posG_3Dumap[entz] = (embed[cc,0],embed[cc,1],embed[cc,2])
        cc += 1

    #--------------------------------------------------------------
    # REST (if genes = G.nodes then rest will be ignored / empty)

    # center for sphere to arrange rest gene-datapoints
    xx=[]
    yy=[]
    zz=[]
    for i in posG_3Dumap.values():
        xx.append(i[0])
        yy.append(i[1])
        zz.append(i[2])

    cx = sum(xx)/len(genes)
    cy = sum(yy)/len(genes)
    cz = sum(zz)/len(genes)

    # generate spherical coordinates for rest genes (without e.g. GO term or Disease Annotation)
    indices = arange(0, len(genes_rest))
    phi = arccos(1 - 2*indices/len(genes_rest)) # 2* --> for both halfs of sphere (upper+lower)
    theta = pi * (1 + 5**0.5) * indices

    xm, ym, zm = max(posG_3Dumap.values())
    r = (math.sqrt((cx - xm)**2 + (cy - ym)**2 + (cz - zm)**2))+1 # +10 to ensure all colored nodes are within the sphere
    x, y, z = cx+r*cos(theta) * sin(phi),cy+r*sin(theta) * sin(phi), cz+r*cos(phi)

    rest_points = []
    for i,j,k in zip(x,y,z):
        rest_points.append((i,j,k))

    posG_rest = dict(zip(genes_rest, rest_points))

    posG_all = {**posG_3Dumap, **posG_rest}
    posG_3D_complete_umap = {key:posG_all[key] for key in G.nodes()}

    # normalize coordinates
    x_list3D = []
    y_list3D = []
    z_list3D = []
    for k,v in posG_3D_complete_umap.items():
        x_list3D.append(v[0])
        y_list3D.append(v[1])
        z_list3D.append(v[2])

    xx_norm3D = sklearn.preprocessing.minmax_scale(x_list3D, feature_range=(0, 1), axis=0, copy=True)
    yy_norm3D = sklearn.preprocessing.minmax_scale(y_list3D, feature_range=(0, 1), axis=0, copy=True)
    zz_norm3D = sklearn.preprocessing.minmax_scale(z_list3D, feature_range=(0, 1), axis=0, copy=True)

    xx_norm3D_final=[]
    for i in xx_norm3D:
        xx_norm3D_final.append(round(i,10))

    yy_norm3D_final=[]
    for i in yy_norm3D:
        yy_norm3D_final.append(round(i,10))

    zz_norm3D_final=[]
    for i in zz_norm3D:
        zz_norm3D_final.append(round(i,10))

    posG_3D_complete_umap_norm = dict(zip(list(G.nodes()), zip(xx_norm3D_final,yy_norm3D_final,zz_norm3D_final)))

    return posG_3D_complete_umap_norm



def embed_umap_sphere(Matrix, n_neighbors, spread, min_dist, metric='cosine'):
    '''
    Generate spherical embedding of nodes in matrix input using UMAP.
    Input:
    - Matrix = Feature Matrix with either all or specific  nodes (rows) and features (columns) or symmetric (nodes = rows and columns)
    - n_neighbors/spread/min_dist = floats; UMAP parameters.
    - metric = string; e.g. havervine, euclidean, cosine ,..

    Return sphere embedding.
    '''

    model = umap.UMAP(
        n_neighbors = n_neighbors,
        spread = spread,
        min_dist = min_dist,
        metric = metric)

    sphere_mapper = model.fit(Matrix)

    return sphere_mapper



def get_posG_sphere(l_genes, sphere_mapper):
    '''
    Generate coordinates from embedding.
    Input:
    - l_genes = list of genes
    - sphere_mapper = embedding from UMAP spherical embedding

    Return dictionary with nodes as keys and coordinates as values in 3D.
    '''

    x = np.sin(sphere_mapper.embedding_[:, 0]) * np.cos(sphere_mapper.embedding_[:, 1])
    y = np.sin(sphere_mapper.embedding_[:, 0]) * np.sin(sphere_mapper.embedding_[:, 1])
    z = np.cos(sphere_mapper.embedding_[:, 0])

    posG = {}
    cc = 0
    for entz in l_genes:
        posG[entz] = (x[cc],y[cc], z[cc])
        cc += 1

    return posG


def get_posG_sphere_norm_old(G, l_genes, sphere_mapper, d_param, radius_rest_genes = 20):
    '''
    Generate coordinates from embedding.
    Input:
    - G = Graph
    - DM = matrix
    - sphere_mapper = embedding from UMAP spherical embedding
    - d_param = dictionary with nodes as keys and assigned radius as values
    - radius_rest_genes = int; radius in case of genes e.g. not function associated if genes not all G.nodes()

    Return dictionary with nodes as keys and coordinates as values in 3D.
    '''

    x = np.sin(sphere_mapper.embedding_[:, 0]) * np.cos(sphere_mapper.embedding_[:, 1])
    y = np.sin(sphere_mapper.embedding_[:, 0]) * np.sin(sphere_mapper.embedding_[:, 1])
    z = np.cos(sphere_mapper.embedding_[:, 0])

    genes = []
    for i in l_genes:
        if str(i) in G.nodes() or int(i) in G.nodes():
            genes.append(str(i))

    genes_rest = []
    for g in G.nodes():
        if g not in genes:
            genes_rest.append(g)

    posG_3Dsphere = {}
    cc = 0
    for entz in genes:
        posG_3Dsphere[entz] = (x[cc],y[cc], z[cc])
        cc += 1

    posG_3Dsphere_radius = {}
    for node,rad in d_param.items():
        for k,v in posG_3Dsphere.items():
            if k == node:
                posG_3Dsphere_radius[k] = (v[0]*rad, v[1]*rad, v[2]*rad)

    # generate spherical coordinates for rest genes (without e.g. GO term or Disease Annotation)
    indices = arange(0, len(genes_rest))
    phi = arccos(1 - 2*indices/len(genes_rest))
    theta = pi * (1 + 5**0.5) * indices

    r_rest = radius_rest_genes # radius for rest genes (e.g. if functional layout)
    x, y, z = r_rest*cos(theta) * sin(phi), r_rest*sin(theta) * sin(phi), r_rest*cos(phi)

    rest_points = []
    for i,j,k in zip(x,y,z):
        rest_points.append((i,j,k))

    posG_rest = dict(zip(genes_rest, rest_points))

    posG_all = {**posG_3Dsphere_radius, **posG_rest}
    posG_complete_sphere = {key:posG_all[key] for key in G.nodes()}

    # normalize coordinates
    x_list = []
    y_list = []
    z_list = []
    for k,v in posG_complete_sphere.items():
        x_list.append(v[0])
        y_list.append(v[1])
        z_list.append(v[2])

    xx_norm = sklearn.preprocessing.minmax_scale(x_list, feature_range=(0, 1), axis=0, copy=True)
    yy_norm = sklearn.preprocessing.minmax_scale(y_list, feature_range=(0, 1), axis=0, copy=True)
    zz_norm = sklearn.preprocessing.minmax_scale(z_list, feature_range=(0, 1), axis=0, copy=True)

    posG_complete_sphere_norm = dict(zip(list(G.nodes()), zip(xx_norm,yy_norm,zz_norm)))

    return posG_complete_sphere_norm


# -------------------------------------------------------------------------------------
# P L O T T I N G
# -------------------------------------------------------------------------------------


def get_trace_nodes_3D(posG, info_list, color_list, size, opac=0.9):
    '''
    Get trace of nodes for plotting in 3D.
    Input:
    - posG = dictionary with nodes as keys and coordinates as values.
    - info_list = hover information for each node, e.g. a list sorted according to the initial graph/posG keys
    - color_list = list of colours obtained from any color function (see above sections).
    - opac = transparency of edges e.g. 0.2

    Return a trace for plotly graph objects plot.
    '''

    key_list=list(posG.keys())
    trace = pgo.Scatter3d(x=[posG[key_list[i]][0] for i in range(len(key_list))],
                           y=[posG[key_list[i]][1] for i in range(len(key_list))],
                           z=[posG[key_list[i]][2] for i in range(len(key_list))],
                           mode = 'markers',
                           text = info_list,
                           hoverinfo = 'text',
                           #textposition='middle center',
                           marker = dict(
                color = color_list,
                size = size,
                symbol = 'circle',
                line = dict(width = 1.0,
                        color = color_list),
                opacity = opac,
            ),
        )

    return trace


def get_trace_edges_3D(G, posG, color_list, opac = 0.2, linewidth=0.2):
    '''
    Get trace of edges for plotting in 3D.
    Input:
    - G = Graph
    - posG = dictionary with nodes as keys and coordinates as values.
    - color_list = list of colours obtained from any color function (see above sections).
    - opac = transparency of edges e.g. 0.2

    Return a trace for plotly graph objects plot.
    '''

    edge_x = []
    edge_y = []
    edge_z = []
    for edge in G.edges():
            x0, y0, z0 = posG[edge[0]]
            x1, y1, z1 = posG[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
            edge_z.append(z0)
            edge_z.append(z1)
            edge_z.append(None)


    trace_edges = pgo.Scatter3d(
                            x = edge_x,
                            y = edge_y,
                            z = edge_z,
                            mode = 'lines', hoverinfo='none',
                            line = dict(width = linewidth, color = color_list),
                            opacity = opac
                    )
    return trace_edges


def get_trace_edges_from_genelist3D(l_spec_edges, posG, col, opac=0.2):
    '''
    Get trace of edges for plotting in 3D only for specific edges.
    Input:
    - G = Graph
    - posG = dictionary with nodes as keys and coordinates as values.
    - color = string; specific color to highlight specific edges

    Return a trace of specific edges.
    '''

    edge_x = []
    edge_y = []
    edge_z = []
    for edge in l_spec_edges:
            x0, y0,z0 = posG[edge[0]]
            x1, y1,z1 = posG[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
            edge_z.append(z0)
            edge_z.append(z1)
            edge_z.append(None)

    trace_edges = pgo.Scatter3d(
                        x = edge_x,
                        y = edge_y,
                        z = edge_z,
                        mode = 'lines', hoverinfo='none',
                        line = dict(width = 1.0, color = [col]*len(edge_x)),
                        opacity = opac
                )
    return trace_edges


def get_trace_edges_landscape(x,y,z0,z):
    '''
    Create trace of vertical connecting edges in between node z0 and node z=parameter (e.g.disease count).
    Return trace with edges.
    '''

    Xe = []
    for u in x:
        Xe += [u,u,None]

    Ye = []
    for v in y:
        Ye += [v,v,None]

    Ze = []
    for w in z0:
        for t in z:
            Ze += [w,t,None]

    trace_edge = pgo.Scatter3d(
        x = Xe,
        y = Ye,
        z = Ze,
        mode = 'lines', hoverinfo='none',
        line = dict(width = 3.0, color = 'darkgrey'),
        opacity = 0.5
    )

    return trace_edge


def plot_3D(data, fname, scheme, annotat=None):
    '''
    Create a 3D plot from traces using plotly.
    Input:
    - data = list of traces
    - filename = string
    - scheme = 'light' or 'dark'
    - annotations = None or plotly annotations

    Return plot in 3D and file, saved as html.
    '''

    fig = pgo.Figure()

    for i in data:
        fig.add_trace(i)

    if scheme == 'dark' and annotat==None:
        fig.update_layout(template='plotly_dark', showlegend=False, autosize = True,
                          scene=dict(
                              xaxis_title='',
                              yaxis_title='',
                              zaxis_title='',
                              xaxis=dict(nticks=0,tickfont=dict(
                                    color='black')),
                              yaxis=dict(nticks=0,tickfont=dict(
                                    color='black')),
                              zaxis=dict(nticks=0,tickfont=dict(
                                    color='black')),
                            dragmode="turntable"
                        ))

    elif scheme == 'dark':
        fig.update_layout(template='plotly_dark', showlegend=False, autosize = True,
                                  scene=dict(
                                      xaxis_title='',
                                      yaxis_title='',
                                      zaxis_title='',
                                      xaxis=dict(nticks=0,tickfont=dict(
                                            color='black')),
                                      yaxis=dict(nticks=0,tickfont=dict(
                                            color='black')),
                                      zaxis=dict(nticks=0,tickfont=dict(
                                            color='black')),
                                    dragmode="turntable",
                                    annotations=annotat,
                                ))

    elif scheme == 'light' and annotat==None:
        fig.update_layout(template='none', showlegend=False, width=1200, height=1200,
                          scene=dict(
                              xaxis_title='',
                              yaxis_title='',
                              zaxis_title='',
                              xaxis=dict(nticks=0,tickfont=dict(
                                    color='white')),
                              yaxis=dict(nticks=0,tickfont=dict(
                                    color='white')),
                              zaxis=dict(nticks=0,tickfont=dict(
                                    color='white')),
                            dragmode="turntable",
                        ))

    elif scheme == 'light':
         fig.update_layout(template='none', showlegend=False, width=1200, height=1200,
                          scene=dict(
                              xaxis_title='',
                              yaxis_title='',
                              zaxis_title='',
                              xaxis=dict(nticks=0,tickfont=dict(
                                    color='white')),
                              yaxis=dict(nticks=0,tickfont=dict(
                                    color='white')),
                              zaxis=dict(nticks=0,tickfont=dict(
                                    color='white')),
                            dragmode="turntable",
                            annotations = annotat
                        ))


    return plotly.offline.plot(fig, filename = fname+'.html', auto_open=True)



def plot2D_app(list_of_traces):

    fig = pgo.Figure()
    for i in list_of_traces:
        fig.add_trace(i)

    fig.update_layout(template='plotly_dark', showlegend=False, autosize = True,
                            margin=dict(l=0, r=0, t=0, b=0),
                            xaxis={'showgrid':False, 'showline':False, 'zeroline':False, 'showticklabels':False},
                            yaxis={'showgrid':False, 'showline':False, 'zeroline':False, 'showticklabels':False}

                        )
    return fig



def plot3D_app(list_of_traces):

    fig = pgo.Figure()
    for i in list_of_traces:
        fig.add_trace(i)

    fig.update_layout(
                            margin=dict(l=0, r=0, t=0, b=0),
                            template=None, paper_bgcolor='black', showlegend=False, autosize = True,#width=1600, height=800,
                            scene=dict(
                            xaxis_title='',
                            yaxis_title='',
                            zaxis_title='',
                            xaxis=dict(nticks=0,tickfont=dict(
                                    color='black')),
                            yaxis=dict(nticks=0,tickfont=dict(
                                    color='black')),
                            zaxis=dict(nticks=0,tickfont=dict(
                                    color='black')),
                            dragmode="turntable",
                        ))

    return fig


# -------------------------------------------------------------------------------------
# P L O T   A N N O T A T I O N S
# -------------------------------------------------------------------------------------

def annotation_kmeansclustering(kmeans, centrs, mode):

    # number of genes in each cluster ( used for annotation )
    d_clus_genecount = dict(collections.Counter(kmeans.labels_))
    d_clus_genecount_sort = dict(collections.OrderedDict(sorted(d_clus_genecount.items())))
    l_clus_genecount = list(d_clus_genecount_sort.values())

    # Centers for clusters ( used for annotation )
    x=[]
    y=[]
    z=[]
    for i in centrs:
        x.append(i[0])
        y.append(i[1])
        z.append(i[2])

    if mode == 'light':
        annotations = []
        for i in range(len(x)):
            annot = dict(
                                x=x[i],
                                y=y[i],
                                z=z[i],
                                showarrow=True,
                                text=f'Cluster: {str(i+1)} <br> total: {str(l_clus_genecount[i])}',
                                font=dict(
                                    color="dimgrey",
                                    size=8),
                                xanchor="right",
                                ay=-20,
                                ax=-20,
                                opacity=0.5,
                                arrowhead=0,
                                arrowwidth=0.5,
                                arrowcolor="dimgrey"
                                )
            i=+1
            annotations.append(annot)
        return annotations

    elif mode == 'dark':
        annotations = []
        for i in range(len(x)):
            annot = dict(
                                x=x[i],
                                y=y[i],
                                z=z[i],
                                showarrow=True,
                                text=f'Cluster: {str(i+1)} <br> total: {str(l_clus_genecount[i])}',
                                font=dict(
                                    color="lightgrey",
                                    size=8),
                                xanchor="right",
                                ay=-20,
                                ax=-20,
                                opacity=0.5,
                                arrowhead=0,
                                arrowwidth=0.5,
                                arrowcolor="lightgrey")
            i=+1
            annotations.append(annot)
        return annotations

    else:
        print('Please choose mode by setting mode="light" or "dark".')



def cluster_annotation(d_clusterid_coords, d_genes_per_cluster, mode = 'light'):
    '''
    Add Anntation of clusters to 3D plot.
    Input:
    - d_clusterid_coords = dictionary with cluster id and x,y,z coordinates of cluster center.
    - d_genes_per_cluster = dictionary with cluster id and genes counted per cluster
    - mode = mode of plot (i.e. 'light', 'dark')

    Return Annotations for each cluster.
    '''

    l_clus_genecount = list(d_genes_per_cluster.values())

    x=[]
    y=[]
    z=[]
    for i in d_clusterid_coords.values():
        x.append(i[0])
        y.append(i[1])
        z.append(i[2])

    if mode == 'light':
        annotations = []
        for i in range(len(x)):
            annot = dict(
                                x=x[i],
                                y=y[i],
                                z=z[i],
                                showarrow=True,
                                text=f'Cluster: {str(i+1)} <br> total: {str(l_clus_genecount[i])}',
                                font=dict(
                                    color="dimgrey",
                                    size=8),
                                xanchor="right",
                                ay=-100,
                                ax=-100,
                                opacity=0.5,
                                arrowhead=0,
                                arrowwidth=0.5,
                                arrowcolor="dimgrey"
                                )
            i=+1
            annotations.append(annot)
        return annotations

    elif mode == 'dark':
        annotations = []
        for i in range(len(x)):
            annot = dict(
                                x=x[i],
                                y=y[i],
                                z=z[i],
                                showarrow=True,
                                text=f'Cluster: {str(i+1)} <br> total: {str(l_clus_genecount[i])}',
                                font=dict(
                                    color="lightgrey",
                                    size=8),
                                xanchor="right",
                                ay=-100,
                                ax=-100,
                                opacity=0.5,
                                arrowhead=0,
                                arrowwidth=0.5,
                                arrowcolor="lightgrey")
            i=+1
            annotations.append(annot)
        return annotations

    else:
        print('Please choose mode by setting mode="light" or "dark".')



def genes_annotation(posG_genes, d_genes, mode = 'light'):
    '''
    Add Anntation of genes to 3D plot.
    Input:
    - posG_genes = dictionary with node id and x,y,z coordinates of cluster center.
    - d_genes = dictionary with node id as keys and symbol (gene symbol) as values. Same order as posG_genes
    - mode of plot (i.e. 'light', 'dark')

    Return Annotations for each cluster.
    '''

    gene_sym = list(d_genes.values())

    x = []
    y = []
    z = []
    for k,v in posG_genes.items():
        x.append(v[0])
        y.append(v[1])
        z.append(v[2])

    if mode == 'light':
        annotations = []
        for i in range(len(x)):
            annot = dict(
                                    x=x[i],
                                    y=y[i],
                                    z=z[i],
                                    showarrow=True,
                                    text=f'Gene: {gene_sym[i]}',
                                    font=dict(
                                        color="black",
                                        size=10),
                                    xanchor="right",
                                    ay=-100,
                                    ax=-100,
                                    opacity=0.5,
                                    arrowhead=0,
                                    arrowwidth=0.5,
                                    arrowcolor="dimgrey")
            i=+1
            annotations.append(annot)
        return annotations

    elif mode == 'dark':
        annotations = []
        for i in range(len(x)):
            annot = dict(
                                    x=x[i],
                                    y=y[i],
                                    z=z[i],
                                    showarrow=True,
                                    text=f'Gene: {gene_sym[i]}',
                                    font=dict(
                                        color="white",
                                        size=10),
                                    xanchor="right",
                                    ay=-100,
                                    ax=-100,
                                    opacity=0.5,
                                    arrowhead=0,
                                    arrowwidth=0.5,
                                    arrowcolor="lightgrey")
            i=+1
            annotations.append(annot)
        return annotations

    else:
        print('Please choose mode by setting mode="light" or "dark".')



###############################################################################################################################
###############################################################################################################################
#
#
#   A P P specific
#
#
###############################################################################################################################
###############################################################################################################################




########################################################################################
#
# E X P O R T   C O O R D I N A T E S   F U N C T I O N S
#
########################################################################################


def to_obj(posG):
    
    verts = list(posG.values()) 
    #faces = elist

    obj_file = []
    for item in verts:
        obj_file.append("v {0} {1} {2}\n".format(item[0],item[1],item[2]))

    #for item in faces:
    #    obj_file.append("f {0}/{0} {1}/{1} {2}/{2}\n".format(item[0],item[1],item[0]))

    return obj_file


def make_wireframe_sphere(centre, radius,
                    n_meridians=20, n_circles_latitude=None):

    if n_circles_latitude is None:
        n_circles_latitude = max(n_meridians/2, 4)
    u, v = np.mgrid[0:2*np.pi:n_meridians*1j, 0:np.pi:n_circles_latitude*1j]
    sphere_x = centre[0] + radius * np.cos(u) * np.sin(v)
    sphere_y = centre[1] + radius * np.sin(u) * np.sin(v)
    sphere_z = centre[2] + radius * np.cos(v)
    
    return sphere_x, sphere_y, sphere_z


def export_to_csv2D_app(layout_namespace, posG, colours):
    '''
    Generate csv for upload to VRnetzer plaform for 2D layouts.
    Return dataframe with ID,X,Y,Z,R,G,B,A,layout_namespace.
    '''

    colours_hex2rgb = []
    for j in colours:
        k = hex_to_rgb(j)
        colours_hex2rgb.append(k)

    colours_r = []
    colours_g = []
    colours_b = []
    colours_a = []
    for i in colours_hex2rgb:
        colours_r.append(int(i[0]))#*255)) # colour values should be integers within 0-255
        colours_g.append(int(i[1]))#*255))
        colours_b.append(int(i[2]))#*255))
        colours_a.append(100) # 0-100 shows normal colours in VR, 128-200 is glowing mode

    df_2D = pd.DataFrame(posG).T
    df_2D.columns=['X','Y']
    df_2D['Z'] = 0
    df_2D['R'] = colours_r
    df_2D['G'] = colours_g
    df_2D['B'] = colours_b
    df_2D['A'] = colours_a
    df_2D[layout_namespace] = layout_namespace

    return df_2D


def export_to_csv3D_app(layout_namespace, posG, colours):
    '''
    Generate csv for upload to VRnetzer plaform for 3D layouts.
    Return dataframe with ID,X,Y,Z,R,G,B,A,layout_namespace.
    '''

    colours_hex2rgb = []
    for j in colours:
        k = hex_to_rgb(j)
        colours_hex2rgb.append(k)

    colours_r = []
    colours_g = []
    colours_b = []
    colours_a = []
    for i in colours_hex2rgb:
        colours_r.append(int(i[0]))#*255)) # colour values should be integers within 0-255
        colours_g.append(int(i[1]))#*255))
        colours_b.append(int(i[2]))#*255))
        colours_a.append(100) # 0-100 shows normal colours in VR, 128-200 is glowing mode

    df_3D = pd.DataFrame(posG).T
    df_3D.columns=['X','Y','Z']
    df_3D['R'] = colours_r
    df_3D['G'] = colours_g
    df_3D['B'] = colours_b
    df_3D['A'] = colours_a
    df_3D[layout_namespace] = layout_namespace

    return df_3D


def export_from_import_csv3D_app(layout_namespace, posG, colours):
    '''
    Generate csv for upload to VRnetzer plaform for 3D layouts.
    Return dataframe with ID,X,Y,Z,R,G,B,A,layout_namespace.
    '''
    #colours_r = []
    #colours_g = []
    #colours_b = []
    #colours_a = []
    #for i in colours:
    #    colours_r.append(int(i[0]))#*255)) # colour values should be integers within 0-255
    #    colours_g.append(int(i[1]))#*255))
    #    colours_b.append(int(i[2]))#*255))
    #    colours_a.append(int(i[3])) # 0-100 shows normal colours in VR, 128-200 is glowing mode

    df_3D = pd.DataFrame(posG).T
    df_3D.columns=['X','Y','Z']
    df_3D['R'] = [int(i[0]) for i in colours]
    df_3D['G'] = [int(i[1]) for i in colours]
    df_3D['B'] = [int(i[2]) for i in colours]
    df_3D['A'] = [int(i[3]) for i in colours]

    df_3D[layout_namespace] = layout_namespace
    df_3D['ID'] = list(posG.keys())

    cols = df_3D.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df_3D_final = df_3D[cols]

    return df_3D_final



def graph_to_xgmml(graph_file, graph, graph_name, directed = False):
    """
    Arguments:
    - `graph_file` output network file (file object)
    - `graph`: NetworkX Graph Object
    - `graph_name`: Name of the graph
    - `directed`: is directed or not
    """
    graph_file = []
    #graph_file.write
    graph_file.append("""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<graph directed="{directed}"  xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns="http://www.cs.rpi.edu/XGMML">
 <att name="selected" value="1" type="boolean" />
 <att name="name" value="{0}" type="string"/>
 <att name="shared name" value="{0}" type="string"/>\n""".format(graph_name, directed=(1 if directed else 0)))

    def quote(text):
        """
        Arguments:
        - `text`:
        """

        return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    def write_att_el(k, v, indent_count):
        indentation_string = ''
        for i in range(0, indent_count):
            indentation_string += '  '
        if isinstance(v, int):
            graph_file.append( #write(
                indentation_string +
                '<att name="{}" value="{}" type="integer" />\n'.format(k, v))
        elif isinstance(v, bool):
            graph_file.append( #write(
                indentation_string +
                '<att name="{}" value="{}" type="boolean" />\n'.format(k, v))
        elif isinstance(v, float):
            graph_file.append( #write(
                indentation_string +
                '<att name="{}" value="{}" type="real" />\n'.format(k, v))
        elif hasattr(v, '__iter__'):
            graph_file.append( #write(
                indentation_string + '<att name="{}" type="list">\n'.format(k))
            for item in v:
                write_att_el(k, item, 3)
            graph_file.append( #write(
                indentation_string + '</att>\n')
        else:
            graph_file.append( #write(
                indentation_string +
                '<att name="{}" value="{}" type="string" />\n'.format(k,
                                                                    quote(v)))

    for onenode in graph.nodes(data=True):
        id = onenode[0]
        attr = dict(onenode[1])

        if 'label' in attr:
            label = attr['label']
            del attr['label']
        else:
            label = id

        graph_file.append( #write(
            '  <node id="{id}" label="{label}">\n'.format(id=id, label=label))

        # Add color element
        if 'color' in attr:
            color = attr['color']
            del attr['color']
            graph_file.append( #write(
                '  <graphics fill="{color}" />\n'.format(color=color))

        for k, v in iter(attr.items()):
            write_att_el(k, v, 2)

        graph_file.append( #write(
            '  </node>\n')

    for oneedge in graph.edges(data=True):
        #
        # The spec, http://cgi5.cs.rpi.edu/research/groups/pb/punin/public_html/XGMML/draft-xgmml.html#GlobalA,
        # requires an "id", even for edges. This id is supposed to be unique across the entire document, so it
        # can't be equal to one of the node ids. We're making the assumption that whoever created the graph
        # object knew about and respected the "uniqueness" requirement, and passed a suitable id as the attribute
        # "id" to the edge. If the creator of the graph *didn't* pass a unique id, the best I can come up with at
        # this moment is to just ignore the id requirement entirely.
        #
        if 'id' in oneedge[2]:
            edge_id = oneedge[2].pop("id", None)
            graph_file.append( #write(
                '  <edge id="{}" source="{}" target="{}">\n'.format(
                edge_id, oneedge[0], oneedge[1]))
        else:
            graph_file.append( #write(
                '  <edge source="{}" target="{}">\n'.format(
                oneedge[0], oneedge[1]))

        for k, v in iter(oneedge[2].items()):
            write_att_el(k, v, 2)
        graph_file.append( #write(
            '  </edge>\n')
    graph_file.append( #write
        '</graph>\n')

    return graph_file


########################################################################################
#
#  L A Y O U T   F U N C T I O N S 
#
#
########################################################################################


############################
#      DRAW 2D LAYOUT
############################
def draw_layout_2D(G, posG, l_feat, colours, node_size, edge_opac, edge_width): 
            opacity_nodes = 0.9
            edge_color = '#ffffff'
            umap2D_nodes = get_trace_nodes_2D(posG, l_feat, colours, node_size, opacity_nodes)
            umap2D_edges = get_trace_edges_2D(G, posG, edge_color, opac=edge_opac, linewidth=edge_width)
            umap2D_data = [umap2D_edges, umap2D_nodes]
            fig2D = plot2D_app(umap2D_data)

            return fig2D

############################
#      DRAW 3D LAYOUT
############################
def draw_layout_3D(G, posG, l_feat, colours, node_size, edge_opac, edge_width): 
            opacity_nodes = 0.9
            edge_color = '#ffffff'
            umap3D_nodes = get_trace_nodes_3D(posG, l_feat, colours, node_size, opacity_nodes)
            umap3D_edges = get_trace_edges_3D(G, posG, edge_color, opac=edge_opac, linewidth=edge_width)
            umap3D_data = [umap3D_edges, umap3D_nodes]
            fig3D = plot3D_app(umap3D_data)

            return fig3D


############################
#
#      PORTRAIT 2D
#
############################

def portrait2D_local(G,dimred):

        closeness = nx.closeness_centrality(G)      
        d_clos_unsort  = {}
        for node, cl in sorted(closeness.items(), key = lambda x: x[1], reverse = 0):
            d_clos_unsort [node] = round(cl,4)
        d_clos = {key:d_clos_unsort[key] for key in G.nodes()}
        d_colours = color_nodes_from_dict(G, d_clos, palette = 'YlOrRd')
        colours = list(d_colours.values())
        l_feat = list(G.nodes())

        A = nx.adjacency_matrix(G, nodelist=list(G.nodes()))
        M_adj = A.toarray()
        DM = pd.DataFrame(M_adj)
        DM.index=list(G.nodes())
        DM.columns=list(G.nodes())

        if dimred == 'tsne':

            prplxty = 20 # range: 5-50
            density = 12 # default 12.
            l_rate = 200 # default 200.
            steps = 250 # min 250
            metric = 'cosine'
            
            r_scale = 1.0
            tsne2D = embed_tsne_2D(DM, prplxty, density, l_rate, steps, metric)
            posG = get_posG_2D_norm(G, DM, tsne2D, r_scale)
         
        elif dimred == 'umap':

            n_neighbors = 8
            spread = 1.0
            min_dist = 0.1
            metric='cosine'
            
            r_scale = 1.0
            umap2D = embed_umap_2D(DM, n_neighbors, spread, min_dist, metric)
            posG = get_posG_2D_norm(G, DM, umap2D, r_scale)
            
        return posG, colours,l_feat


def portrait2D_global(G, dimred):

        closeness = nx.closeness_centrality(G)
        d_clos_unsort  = {}
        for node, cl in sorted(closeness.items(), key = lambda x: x[1], reverse = 0):
            d_clos_unsort [node] = round(cl,4)
        d_clos = {key:d_clos_unsort[key] for key in G.nodes()}
        d_colours = color_nodes_from_dict(G, d_clos, palette = 'YlOrRd')
        colours = list(d_colours.values())
        l_feat = list(G.nodes())

        A = nx.adjacency_matrix(G, nodelist=list(G.nodes()))
        DM = pd.DataFrame(rnd_walk_matrix2(A,0.9,1,len(G))).T
        DM.index=list(G.nodes())
        DM.columns=list(G.nodes())

        if dimred == 'tsne':

            prplxty = 20 # range: 5-50
            density = 12 # default 12.
            l_rate = 200 # default 200.
            steps = 250 # min 250
            metric = 'cosine'
            
            r_scale = 1.2
            tsne2D = embed_tsne_2D(DM, prplxty, density, l_rate, steps, metric)
            posG = get_posG_2D_norm(G, DM, tsne2D, r_scale)
         
        elif dimred == 'umap':

            n_neighbors = 8
            spread = 1.0
            min_dist = 0.1
            metric='cosine'
            
            r_scale = 2.0
            umap2D = embed_umap_2D(DM, n_neighbors, spread, min_dist, metric)
            posG = get_posG_2D_norm(G, DM, umap2D, r_scale)
            
        return posG, colours,l_feat



def portrait2D_importance(G, dimred):

        closeness = nx.closeness_centrality(G)
        d_clos_unsort  = {}
        for node, cl in sorted(closeness.items(), key = lambda x: x[1], reverse = 0):
            d_clos_unsort [node] = round(cl,4)
        d_clos = {key:d_clos_unsort[key] for key in G.nodes()}
        d_colours = color_nodes_from_dict(G, d_clos, palette = 'YlOrRd')
        colours = list(d_colours.values())
        l_feat = list(G.nodes())

        colours = list(d_colours.values())
        l_feat = list(G.nodes())
        d_degs = dict(G.degree())
        betweens = nx.betweenness_centrality(G)
        d_betw = {}
        for node, be in sorted(betweens.items(), key = lambda x: x[1], reverse = 1):
            d_betw[node] = round(be,4)
        d_degs_sorted = {key:d_degs[key] for key in sorted(d_degs.keys())}
        d_clos_sorted = {key:d_clos[key] for key in sorted(d_clos.keys())}
        d_betw_sorted = {key:d_betw[key] for key in sorted(d_betw.keys())}
        feature_dict = dict(zip(d_degs_sorted.keys(), zip(d_degs_sorted.values(), d_clos_sorted.values(), d_betw_sorted.values())))
        feature_dict_sorted = {key:feature_dict[key] for key in G.nodes()}
        DM = pd.DataFrame.from_dict(feature_dict_sorted, orient = 'index', columns = ['degs', 'clos', 'betw'])

        if dimred == 'tsne':

            prplxty = 20 # range: 5-50
            density = 12 # default 12.
            l_rate = 200 # default 200.
            steps = 250 # min 250
            metric = 'cosine'
            tsne = embed_tsne_2D(DM, prplxty, density, l_rate, steps, metric)
            posG = get_posG_2D_norm(G, DM, tsne)
         
        elif dimred == 'umap':

            n_neighbors = 8
            spread = 1.0
            min_dist = 0.1
            metric='cosine'
            embed = embed_umap_2D(DM,n_neighbors,spread,min_dist,metric)
            posG = get_posG_2D_norm(G,DM,embed)
            
        return posG, colours,l_feat


def portrait2D_functional(G, FM, dimred = 'umap'):
        
        graph_nodes = [str(i) for i in list(G.nodes())]

        pal_rgb = sns.color_palette("husl", len(FM.columns))
        pal = list(map(matplotlib.colors.rgb2hex, pal_rgb))
        
        colors_assigned_unsorted = {}
        d_feat = {}
        for j,row in enumerate(FM.iterrows()):
            # row[0] is the row = index of Graph nodes
            # row[1][0] is the values in the feature columns 0
            # row[1][1] is the values in the feature columns 1
            # row[1][2] is the values in the feature columns 2
            for k,v in enumerate(row[1]):
                if v == 1:
                    feature = 'group: '+str(k)
                    colors_assigned_unsorted[str(row[0])] = pal[k]
                    d_feat[str(row[0])] = feature
                else:
                    pass
        
        cols_merged = {}
        for node in G.nodes():
            for n,c in colors_assigned_unsorted.items():
                if node not in colors_assigned_unsorted.keys():
                    cols_merged[node] = '#d3d3d3' 
                else:
                    cols_merged[n] = c

        d_feat_merged = {}
        for node in G.nodes():
            for n,f in d_feat.items():
                if node not in d_feat.keys():
                    d_feat_merged[node] = 'NA'
                else:
                    d_feat_merged[n] = f
                    
        colors_assigned_sort = {key:cols_merged[key] for key in graph_nodes}
        colours = list(colors_assigned_sort.values())
        
        d_feat_sort = {key:d_feat_merged[key] for key in graph_nodes}
        l_feat = list(d_feat_sort.values())
        
        DM = FM 

        if dimred == 'tsne':

            prplxty = 20 # range: 5-50
            density = 12 # default 12.
            l_rate = 200 # default 200.
            steps = 250 # min 250
            metric = 'cosine'
            tsne = embed_tsne_2D(DM, prplxty, density, l_rate, steps, metric)
            posG = get_posG_2D_norm(G, DM, tsne)
         
        elif dimred == 'umap':

            n_neighbors = 8
            spread = 1.0
            min_dist = 0.1
            metric='cosine'
            embed = embed_umap_2D(DM,n_neighbors,spread,min_dist,metric)
            posG = get_posG_2D_norm(G,DM,embed)
        
        return posG, colours,l_feat


def layout_functional_umap(G, Matrix,dim,n_neighbors=8, spread=1.0, min_dist=0.0, metric='cosine',r_scale = 1.2):
    
    if dim == 2:
        umap2D = embed_umap_2D(Matrix, n_neighbors, spread, min_dist, metric)
        posG = get_posG_2D_norm(G, Matrix, umap2D,r_scale)
        
        return posG
    
    elif dim == 3: 
        umap_3D = embed_umap_3D(Matrix, n_neighbors, spread, min_dist, metric)
        posG = get_posG_3D_norm(G, Matrix, umap_3D,r_scale)

        return posG
        
    else:
        print('Please choose dimensions, by either setting dim=2 or dim=3.')


############################
#
#      PORTRAIT 3D
#
############################

def portrait3D_local(G, dimred):

        closeness = nx.closeness_centrality(G)
        d_clos_unsort  = {}
        for node, cl in sorted(closeness.items(), key = lambda x: x[1], reverse = 0):
            d_clos_unsort [node] = round(cl,4)
        d_clos = {key:d_clos_unsort[key] for key in G.nodes()}
        d_colours = color_nodes_from_dict(G, d_clos, palette = 'YlOrRd')
        colours = list(d_colours.values())
        l_feat = list(G.nodes())

        A = nx.adjacency_matrix(G, nodelist=list(G.nodes()))
        M_adj = A.toarray()
        DM = pd.DataFrame(M_adj)
        DM.index=list(G.nodes())
        DM.columns=list(G.nodes())

        if dimred == 'tsne':

            prplxty = 20 # range: 5-50
            density = 12 # default 12.
            l_rate = 200 # default 200.
            steps = 250 # min 250
            metric = 'cosine'
            tsne_3D = embed_tsne_3D(DM, prplxty, density, l_rate, steps, metric)
            posG_3D = get_posG_3D_norm(G, DM, tsne_3D)
         
        elif dimred == 'umap':

            n_neighbors = 8
            spread = 1.0
            min_dist = 0.1
            metric='cosine'
            embed3D = embed_umap_3D(DM,n_neighbors,spread,min_dist,metric)
            posG_3D = get_posG_3D_norm(G,DM,embed3D)
            
        return posG_3D, colours,l_feat


def portrait3D_global(G,dimred): #,node_size = 1.5,edge_width = 0.8, edge_opac = 0.5):
       
        #closeness = nx.closeness_centrality(G)
        closeness = nx.degree_centrality(G)
        d_clos_unsort  = {}
        for node, cl in sorted(closeness.items(), key = lambda x: x[1], reverse = 0):
            d_clos_unsort [node] = round(cl,4)
        d_clos = {key:d_clos_unsort[key] for key in G.nodes()}
        d_colours = color_nodes_from_dict(G, d_clos, palette = 'YlOrRd')
        colours = list(d_colours.values())
        l_feat = list(G.nodes())

        A = nx.adjacency_matrix(G, nodelist=list(G.nodes()))
        DM_m = pd.DataFrame(rnd_walk_matrix2(A,0.9,1,len(G))).T
        DM_m.index=list(G.nodes())
        DM_m.columns=list(G.nodes())

        if dimred == 'tsne':

            prplxty = 20 # range: 5-50
            density = 12 # default 12.
            l_rate = 200 # default 200.
            steps = 250 # min 250
            metric = 'cosine'
            tsne_3D = embed_tsne_3D(DM_m, prplxty, density, l_rate, steps, metric)
            posG_3D = get_posG_3D_norm(G, DM_m, tsne_3D)
         
        elif dimred == 'umap':

            n_neighbors = 8
            spread = 1
            min_dist = 0.1
            metric='cosine'
            embed3D = embed_umap_3D(DM_m,n_neighbors,spread,min_dist,metric)
            posG_3D = get_posG_3D_norm(G,DM_m,embed3D)
            
        return posG_3D, colours,l_feat

def portrait3D_importance(G, dimred):

        closeness = nx.closeness_centrality(G)
        d_clos_unsort  = {}
        for node, cl in sorted(closeness.items(), key = lambda x: x[1], reverse = 0):
            d_clos_unsort [node] = round(cl,4)
        d_clos = {key:d_clos_unsort[key] for key in G.nodes()}
        d_colours = color_nodes_from_dict(G, d_clos, palette = 'YlOrRd')
        colours = list(d_colours.values())
        l_feat = list(G.nodes())

        colours = list(d_colours.values())
        node_size = 1.5
        l_feat = list(G.nodes())
        d_degs = dict(G.degree())
        betweens = nx.betweenness_centrality(G)
        d_betw = {}
        for node, be in sorted(betweens.items(), key = lambda x: x[1], reverse = 1):
            d_betw[node] = round(be,4)
        d_degs_sorted = {key:d_degs[key] for key in sorted(d_degs.keys())}
        d_clos_sorted = {key:d_clos[key] for key in sorted(d_clos.keys())}
        d_betw_sorted = {key:d_betw[key] for key in sorted(d_betw.keys())}
        feature_dict = dict(zip(d_degs_sorted.keys(), zip(d_degs_sorted.values(), d_clos_sorted.values(), d_betw_sorted.values())))
        feature_dict_sorted = {key:feature_dict[key] for key in G.nodes()}
        DM = pd.DataFrame.from_dict(feature_dict_sorted, orient = 'index', columns = ['degs', 'clos', 'betw'])

        if dimred == 'tsne':

            prplxty = 20 # range: 5-50
            density = 12 # default 12.
            l_rate = 200 # default 200.
            steps = 250 # min 250
            metric = 'cosine'
            tsne_3D = embed_tsne_3D(DM, prplxty, density, l_rate, steps, metric)
            posG_3D = get_posG_3D_norm(G, DM, tsne_3D)
         
        elif dimred == 'umap':

            n_neighbors = 20
            spread = 0.9
            min_dist = 0
            metric='cosine'
            embed3D = embed_umap_3D(DM,n_neighbors,spread,min_dist,metric)
            posG_3D = get_posG_3D_norm(G,DM,embed3D)
            
        return posG_3D, colours,l_feat


def portrait3D_functional(G, FM, dimred = 'umap'):
        
        graph_nodes = [str(i) for i in list(G.nodes())]

        pal_rgb = sns.color_palette("husl", len(FM.columns))
        pal = list(map(matplotlib.colors.rgb2hex, pal_rgb))
        
        colors_assigned_unsorted = {}
        d_feat = {}
        for j,row in enumerate(FM.iterrows()):
            # row[0] is the row = index of Graph nodes
            # row[1][0] is the values in the feature columns 0
            # row[1][1] is the values in the feature columns 1
            # row[1][2] is the values in the feature columns 2
            for k,v in enumerate(row[1]):
                if v == 1:
                    feature = 'group: '+str(k)
                    colors_assigned_unsorted[str(row[0])] = pal[k]
                    d_feat[str(row[0])] = feature
                else:
                    pass
        
        cols_merged = {}
        for node in G.nodes():
            for n,c in colors_assigned_unsorted.items():
                if node not in colors_assigned_unsorted.keys():
                    cols_merged[node] = '#d3d3d3' 
                else:
                    cols_merged[n] = c

        d_feat_merged = {}
        for node in G.nodes():
            for n,f in d_feat.items():
                if node not in d_feat.keys():
                    d_feat_merged[node] = 'NA'
                else:
                    d_feat_merged[n] = f
                    
        colors_assigned_sort = {key:cols_merged[key] for key in graph_nodes}
        colours = list(colors_assigned_sort.values())
        
        d_feat_sort = {key:d_feat_merged[key] for key in graph_nodes}
        l_feat = list(d_feat_sort.values())
        
        DM = FM  

        if dimred == 'tsne':

            prplxty = 20 # range: 5-50
            density = 12 # default 12.
            l_rate = 200 # default 200.
            steps = 250 # min 250
            metric = 'cosine'
            tsne = embed_tsne_3D(DM, prplxty, density, l_rate, steps, metric)
            posG = get_posG_3D_norm(G, DM, tsne)
         
        elif dimred == 'umap':

            n_neighbors = 8
            spread = 1.0
            min_dist = 0.1
            metric='cosine'
            embed = embed_umap_3D(DM,n_neighbors,spread,min_dist,metric)
            posG = get_posG_3D_norm(G,DM,embed)
        
        return posG, colours,l_feat



############################
#
#      TOPOGRAPHIC
#
############################

def topographic_local(G, z_list, dimred):
        
        closeness = nx.closeness_centrality(G)      
        d_clos_unsort  = {}
        for node, cl in sorted(closeness.items(), key = lambda x: x[1], reverse = 0):
            d_clos_unsort [node] = round(cl,4)
        d_clos = {key:d_clos_unsort[key] for key in G.nodes()}
        d_colours = color_nodes_from_dict(G, d_clos, palette = 'YlOrRd')
        colours = list(d_colours.values())
        l_feat = list(G.nodes())

        A = nx.adjacency_matrix(G, nodelist=list(G.nodes()))
        M_adj = A.toarray()
        DM = pd.DataFrame(M_adj)
        DM.index=list(G.nodes())
        DM.columns=list(G.nodes())

        z_list_norm = sklearn.preprocessing.minmax_scale(z_list, feature_range=(0, 1.0), axis=0, copy=True)
        r_scale=1.2

        if dimred == 'tsne': 
            prplxty = 20 # range: 5-50
            density = 12 # default 12.
            l_rate = 200 # default 200.
            steps = 250 # min 250
            metric = 'cosine'

            tsne = embed_tsne_2D(DM, prplxty, density, l_rate, steps, metric)
            posG_ = get_posG_2D_norm(G, DM, tsne, r_scale)
            
        elif dimred == 'umap':   
            n_neighbors = 8
            spread = 1.0
            min_dist = 0.1
            metric='cosine'

            umap = embed_umap_2D(DM, n_neighbors, spread, min_dist, metric)
            posG_ = get_posG_2D_norm(G, DM, umap, r_scale)
            
        posG = {}
        cc = 0
        for k,v in posG_.items():
            posG[k] = (v[0],v[1],z_list_norm[cc])
            cc+=1

        return posG, colours,l_feat


def topographic_global(G, z_list, dimred):

        closeness = nx.closeness_centrality(G)      
        d_clos_unsort  = {}
        for node, cl in sorted(closeness.items(), key = lambda x: x[1], reverse = 0):
            d_clos_unsort [node] = round(cl,4)
        d_clos = {key:d_clos_unsort[key] for key in G.nodes()}
        d_colours = color_nodes_from_dict(G, d_clos, palette = 'YlOrRd')
        colours = list(d_colours.values())
        l_feat = list(G.nodes())

        A = nx.adjacency_matrix(G, nodelist=list(G.nodes()))
        DM = pd.DataFrame(rnd_walk_matrix2(A,0.9,1,len(G))).T
        DM.index=list(G.nodes())
        DM.columns=list(G.nodes())

        z_list_norm = sklearn.preprocessing.minmax_scale(z_list, feature_range=(0, 1.0), axis=0, copy=True)
        r_scale=1.2

        if dimred == 'tsne': 
            prplxty = 20 # range: 5-50
            density = 12 # default 12.
            l_rate = 200 # default 200.
            steps = 250 # min 250
            metric = 'cosine'

            tsne = embed_tsne_2D(DM, prplxty, density, l_rate, steps, metric)
            posG_ = get_posG_2D_norm(G, DM, tsne, r_scale)
            
        elif dimred == 'umap':  
            n_neighbors = 8
            spread = 1.0
            min_dist = 0.1
            metric='cosine'

            umap = embed_umap_2D(DM, n_neighbors, spread, min_dist, metric)
            posG_ = get_posG_2D_norm(G, DM, umap, r_scale)
            
        posG = {}
        cc = 0
        for k,v in posG_.items():
            posG[k] = (v[0],v[1],z_list_norm[cc])
            cc+=1

        return posG, colours,l_feat
 

def topographic_importance(G, z_list, dimred):

        closeness = nx.closeness_centrality(G)      
        d_clos_unsort  = {}
        for node, cl in sorted(closeness.items(), key = lambda x: x[1], reverse = 0):
            d_clos_unsort [node] = round(cl,4)
        d_clos = {key:d_clos_unsort[key] for key in G.nodes()}
        d_colours = color_nodes_from_dict(G, d_clos, palette = 'YlOrRd')
        colours = list(d_colours.values())
        l_feat = list(G.nodes())

        d_degs = dict(G.degree())
        betweens = nx.betweenness_centrality(G)
        d_betw = {}
        for node, be in sorted(betweens.items(), key = lambda x: x[1], reverse = 1):
            d_betw[node] = round(be,4)
        d_degs_sorted = {key:d_degs[key] for key in sorted(d_degs.keys())}
        d_clos_sorted = {key:d_clos[key] for key in sorted(d_clos.keys())}
        d_betw_sorted = {key:d_betw[key] for key in sorted(d_betw.keys())}
        feature_dict = dict(zip(d_degs_sorted.keys(), zip(d_degs_sorted.values(), d_clos_sorted.values(), d_betw_sorted.values())))
        feature_dict_sorted = {key:feature_dict[key] for key in G.nodes()}
        DM = pd.DataFrame.from_dict(feature_dict_sorted, orient = 'index', columns = ['degs', 'clos', 'betw'])
       
        z_list_norm = sklearn.preprocessing.minmax_scale(z_list, feature_range=(0, 1.0), axis=0, copy=True)
        r_scale=1.2

        if dimred == 'tsne': 
            prplxty = 20 # range: 5-50
            density = 12 # default 12.
            l_rate = 200 # default 200.
            steps = 250 # min 250
            metric = 'cosine'

            tsne = embed_tsne_2D(DM, prplxty, density, l_rate, steps, metric)
            posG_ = get_posG_2D_norm(G, DM, tsne, r_scale)
            
        elif dimred == 'umap':  
            n_neighbors = 8
            spread = 1.0
            min_dist = 0.1
            metric='cosine'
            
            umap = embed_umap_2D(DM, n_neighbors, spread, min_dist, metric)
            posG_ = get_posG_2D_norm(G, DM, umap, r_scale)
            
        posG = {}
        cc = 0
        for k,v in posG_.items():
            posG[k] = (v[0],v[1],z_list_norm[cc])
            cc+=1

        return posG, colours,l_feat


def topographic_functional(G, FM, dimred):
        
        r_scale=1.2

        graph_nodes = [str(i) for i in list(G.nodes())]

        pal_rgb = sns.color_palette("husl", len(FM.columns))
        pal = list(map(matplotlib.colors.rgb2hex, pal_rgb))

        colors_assigned_unsorted = {}
        d_feat = {}
        d_z = {}
        for j,row in enumerate(FM.iterrows()):
            # row[0] is the row = index of Graph nodes
            # row[1][0] is the values in the feature columns 0
            # row[1][1] is the values in the feature columns 1
            # row[1][2] is the values in the feature columns 2
            for k,v in enumerate(row[1]):
                if v == 1:
                    feature = 'group: '+str(k)
                    colors_assigned_unsorted[str(row[0])] = pal[k]
                    d_feat[str(row[0])] = feature
                    d_z[str(row[0])] = k+5
                else:
                    pass

        cols_merged = {}
        for node in G.nodes():
            for n,c in colors_assigned_unsorted.items():
                if node not in colors_assigned_unsorted.keys():
                    cols_merged[node] = '#d3d3d3' 
                else:
                    cols_merged[n] = c

        d_feat_merged = {}
        for node in G.nodes():
            for n,f in d_feat.items():
                if node not in d_feat.keys():
                    d_feat_merged[node] = 'NA'
                else:
                    d_feat_merged[n] = f

        d_z_merged = {}
        for node in G.nodes():
            for n,z in d_z.items():
                if node not in d_z.keys():
                    d_z_merged[node] = 0
                else:
                    d_z_merged[n] = z

        colors_assigned_sort = {key:cols_merged[key] for key in graph_nodes}
        colours = list(colors_assigned_sort.values())
        
        d_z_sort = {key:d_z_merged[key] for key in graph_nodes}
        z_list = list(d_z_sort.values())
        d_feat_sort = {key:d_feat_merged[key] for key in graph_nodes}
        l_feat = list(d_feat_sort.values())
        
        z_list_norm = sklearn.preprocessing.minmax_scale(z_list, feature_range=(0, 1.0), axis=0, copy=True)

        DM = FM 

        if dimred == 'tsne': 
            prplxty = 20 # range: 5-50
            density = 12 # default 12.
            l_rate = 200 # default 200.
            steps = 250 # min 250
            metric = 'cosine'

            tsne = embed_tsne_2D(DM, prplxty, density, l_rate, steps, metric)
            posG_ = get_posG_2D_norm(G, DM, tsne, r_scale)
            
        elif dimred == 'umap':  
            n_neighbors = 8
            spread = 1.0
            min_dist = 0.1
            metric='cosine'

            umap = embed_umap_2D(DM, n_neighbors, spread, min_dist, metric)
            posG_ = get_posG_2D_norm(G, DM, umap, r_scale)
            
        posG = {}
        cc = 0
        for k,v in posG_.items():
            posG[k] = (v[0],v[1],z_list_norm[cc])
            cc+=1

        return posG, colours,l_feat



############################
#
#      GEODESIC
#
############################

def geodesic_local(G, dict_radius): #, int_restradius):

        closeness = nx.closeness_centrality(G)      
        d_clos_unsort  = {}
        for node, cl in sorted(closeness.items(), key = lambda x: x[1], reverse = 0):
            d_clos_unsort [node] = round(cl,4)
        d_clos = {key:d_clos_unsort[key] for key in G.nodes()}
        d_colours = color_nodes_from_dict(G, d_clos, palette = 'YlOrRd')
        colours = list(d_colours.values())
        l_feat = list(G.nodes())

        A = nx.adjacency_matrix(G, nodelist=list(G.nodes()))
        M_adj = A.toarray()
        DM = pd.DataFrame(M_adj)
        DM.index=list(G.nodes())
        DM.columns=list(G.nodes())

        n_neighbors = 8
        spread = 1.0
        min_dist = 0.1
        metric='cosine'
        genes = list(G.nodes())
        umap_sphere = embed_umap_sphere(DM, n_neighbors, spread, min_dist, metric)
        posG = get_posG_sphere_norm(G, DM, umap_sphere, dict_radius)#, int_rest_radius

        return posG, colours, l_feat

def geodesic_global(G,dict_radius):

        closeness = nx.closeness_centrality(G)      
        d_clos_unsort  = {}
        for node, cl in sorted(closeness.items(), key = lambda x: x[1], reverse = 0):
            d_clos_unsort [node] = round(cl,4)
        d_clos = {key:d_clos_unsort[key] for key in G.nodes()}
        d_colours = color_nodes_from_dict(G, d_clos, palette = 'YlOrRd')
        colours = list(d_colours.values())
        l_feat = list(G.nodes())

        A = nx.adjacency_matrix(G, nodelist=list(G.nodes()))
        DM = pd.DataFrame(rnd_walk_matrix2(A,0.9,1,len(G))).T
        DM.index=list(G.nodes())
        DM.columns=list(G.nodes())

        n_neighbors = 8
        spread = 1.0
        min_dist = 0.1
        metric='cosine'
        genes = list(G.nodes())
        umap_sphere = embed_umap_sphere(DM, n_neighbors, spread, min_dist, metric)
        posG = get_posG_sphere_norm(G, DM, umap_sphere, dict_radius)#, int_rest_radius

        return posG, colours, l_feat


def geodesic_importance(G,dict_radius):

        closeness = nx.closeness_centrality(G)      
        d_clos_unsort  = {}
        for node, cl in sorted(closeness.items(), key = lambda x: x[1], reverse = 0):
            d_clos_unsort [node] = round(cl,4)
        d_clos = {key:d_clos_unsort[key] for key in G.nodes()}
        d_colours = color_nodes_from_dict(G, d_clos, palette = 'YlOrRd')
        colours = list(d_colours.values())
        l_feat = list(G.nodes())

        d_degs = dict(G.degree())
        betweens = nx.betweenness_centrality(G)
        d_betw = {}
        for node, be in sorted(betweens.items(), key = lambda x: x[1], reverse = 1):
            d_betw[node] = round(be,4)
        d_degs_sorted = {key:d_degs[key] for key in sorted(d_degs.keys())}
        d_clos_sorted = {key:d_clos[key] for key in sorted(d_clos.keys())}
        d_betw_sorted = {key:d_betw[key] for key in sorted(d_betw.keys())}
        feature_dict = dict(zip(d_degs_sorted.keys(), zip(d_degs_sorted.values(), d_clos_sorted.values(), d_betw_sorted.values())))
        feature_dict_sorted = {key:feature_dict[key] for key in G.nodes()}
        DM = pd.DataFrame.from_dict(feature_dict_sorted, orient = 'index', columns = ['degs', 'clos', 'betw'])

        n_neighbors = 8
        spread = 1.0
        min_dist = 0.1
        metric='cosine'
        genes = list(G.nodes())
        umap_sphere = embed_umap_sphere(DM, n_neighbors, spread, min_dist, metric)
        posG = get_posG_sphere_norm(G, DM, umap_sphere, dict_radius)#, int_rest_radius

        return posG, colours, l_feat


def geodesic_functional(G, FM):
        
        graph_nodes = [str(i) for i in list(G.nodes())]

        pal_rgb = sns.color_palette("husl", len(FM.columns))
        pal = list(map(matplotlib.colors.rgb2hex, pal_rgb))

        colors_assigned_unsorted = {}
        d_feat = {}
        d_radius_unsort = {}
        for j,row in enumerate(FM.iterrows()):
            # row[0] is the row = index of Graph nodes
            # row[1][0] is the values in the feature columns 0
            # row[1][1] is the values in the feature columns 1
            # row[1][2] is the values in the feature columns 2
            for k,v in enumerate(row[1]):
                if v == 1:
                    feature = 'group: '+str(k)
                    colors_assigned_unsorted[str(row[0])] = pal[k]
                    d_feat[str(row[0])] = feature
                    d_radius_unsort[str(row[0])] = (k+1)**2
                else:
                    pass

        cols_merged = {}
        for node in G.nodes():
            for n,c in colors_assigned_unsorted.items():
                if node not in colors_assigned_unsorted.keys():
                    cols_merged[node] = '#d3d3d3' 
                else:
                    cols_merged[n] = c

        d_feat_merged = {}
        for node in G.nodes():
            for n,f in d_feat.items():
                if node not in d_feat.keys():
                    d_feat_merged[node] = 'NA'
                else:
                    d_feat_merged[n] = f

        d_radius_merged = {}
        for node in G.nodes():
            for n,r in d_radius_unsort.items():
                if node not in d_radius_unsort.keys():
                    d_radius_merged[node] = 1
                else:
                    d_radius_merged[n] = r

        colors_assigned_sort = {key:cols_merged[key] for key in graph_nodes}
        colours = list(colors_assigned_sort.values())
        
        d_feat_sort = {key:d_feat_merged[key] for key in graph_nodes}
        l_feat = list(d_feat_sort.values())
        
        d_radius_sort = {key:d_radius_merged[key] for key in graph_nodes}

        DM = FM 

        n_neighbors = 8
        spread = 1.0
        min_dist = 0.1
        metric='cosine'
        #genes = list(G.nodes())
        umap_sphere = embed_umap_sphere(DM, n_neighbors, spread, min_dist, metric)
        posG = get_posG_sphere_norm(G, DM, umap_sphere, d_radius_sort)
        return posG, colours, l_feat


def synthetic_featurematrix(G):

    graph_nodes = [str(i) for i in list(G.nodes())]     
                
    scale = 1
    val = 0
    rows = len(list(G.nodes()))
    feat_one = [(val) if i%3 else (scale) for i in range(rows)]
    feat_two = [(val) if i%2 or feat_one[i]==scale in feat_one else (scale) for i in range(rows)]
    feat_three = [(scale) if feat_one[i]==val and feat_two[i]==val else val for i in range(rows)]
    feat_matrix = np.vstack((feat_one,feat_two,feat_three))
                                    
    FM = pd.DataFrame(feat_matrix).T
    FM.index = graph_nodes

    return FM

