
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
import matplotlib.pyplot as plt

import networkx as nx
from networkx.algorithms.flow import shortest_augmenting_path
from networkx.generators.degree_seq import expected_degree_graph
from networkx.algorithms.community import greedy_modularity_communities
from networkx.readwrite.adjlist import parse_adjlist
from networkx.readwrite.edgelist import parse_edgelist

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

def import_vrnetzer_csv(G,file):

    edge_width = 0.8
    edge_opac = 0.05
    edge_colordark = '#666666'
    node_size = 1.5
    #nodesglow_diameter = 20.0
    #nodesglow_transparency = 0.05 # 0.01 

    df = pd.read_csv(file,header=None)
    df.columns = ['id','x','y','z','r','g','b','a','namespace']

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

    return fig 



############################
#
#      PORTRAIT 2D 
#
############################

def portrait2D_local(G):

        n_neighbors = 20 
        spread = 0.9
        min_dist = 0
        metric='cosine'

        edge_width = 0.8
        edge_opac = 0.5
        edge_colordark = '#d3d3d3'
        node_edge_col = '#696969'
        node_size = 5
        opacity_nodes = 0.9
        nodesglow_diameter = 20.0
        nodesglow_transparency = 0.01 # 0.05

        closeness = nx.closeness_centrality(G)
        d_clos_unsort  = {}
        for node, cl in sorted(closeness.items(), key = lambda x: x[1], reverse = 0):
            d_clos_unsort [node] = round(cl,4)  
        col_pal = 'viridis'
        d_clos = {key:d_clos_unsort[key] for key in G.nodes()}
        d_nodecol = d_clos
        d_colours = color_nodes_from_dict(G, d_nodecol, palette = col_pal)
        colours = list(d_colours.values())
        l_feat = list(G.nodes())

        A = nx.adjacency_matrix(G, nodelist=list(G.nodes()))
        M_adj = A.toarray()
        DM_adj = pd.DataFrame(M_adj)
        DM_adj.index=list(G.nodes())
        DM_adj.columns=list(G.nodes())

        r_scale = 1.2
        umap2D = embed_umap_2D(DM_adj, n_neighbors, spread, min_dist, metric)
        posG = get_posG_2D_norm(G, DM_adj, umap2D, r_scale)      

        nodes = get_trace_nodes_2D(posG, l_feat, colours, node_size) #, node_opac)
        nodes_glow = get_trace_nodes_2D(posG, None, colours, nodesglow_diameter, nodesglow_transparency)

        edges = get_trace_edges_2D(G, posG, edge_colordark, opac = edge_opac)
        data = [nodes_glow, edges, nodes]
        fig = plot2D_app(data)

        return fig , posG, colours

def portrait2D_global(G):

        n_neighbors = 20 
        spread = 0.9
        min_dist = 0
        metric='cosine'

        edge_width = 0.8
        edge_opac = 0.5
        edge_colordark = '#666666'
        node_edge_col = '#696969'
        node_size = 5
        opacity_nodes = 0.9
        nodesglow_diameter = 20.0
        nodesglow_transparency = 0.01 # 0.05

        closeness = nx.closeness_centrality(G)
        d_clos_unsort  = {}
        for node, cl in sorted(closeness.items(), key = lambda x: x[1], reverse = 0):
            d_clos_unsort [node] = round(cl,4)  
        col_pal = 'viridis'
        d_clos = {key:d_clos_unsort[key] for key in G.nodes()}
        d_nodecol = d_clos
        d_colours = color_nodes_from_dict(G, d_nodecol, palette = col_pal)
        colours = list(d_colours.values())
        node_size = 1.5
        l_feat = list(G.nodes())

        A = nx.adjacency_matrix(G, nodelist=list(G.nodes()))
        DM_m = pd.DataFrame(rnd_walk_matrix2(A,0.9,1,len(G))).T
        DM_m.index=list(G.nodes())
        DM_m.columns=list(G.nodes())
  
        r_scale = 1.2
        umap2D = embed_umap_2D(DM_m, n_neighbors, spread, min_dist, metric)
        posG = get_posG_2D_norm(G, DM_m, umap2D, r_scale)      

        nodes = get_trace_nodes_2D(posG, l_feat, colours, node_size) #, node_opac)
        nodes_glow = get_trace_nodes_2D(posG, None, colours, nodesglow_diameter, nodesglow_transparency)
        edges = get_trace_edges_2D(G, posG, edge_colordark, opac = edge_opac)
        data = [nodes_glow, edges, nodes]
        fig = plot2D_app(data)

        return fig , posG, colours

def portrait2D_importance(G):

        n_neighbors = 20 
        spread = 0.9
        min_dist = 0
        metric='cosine'

        edge_width = 0.8
        edge_opac = 0.5
        edge_colordark = '#666666'
        node_edge_col = '#696969'
        node_size = 5
        opacity_nodes = 0.9
        nodesglow_diameter = 20.0
        nodesglow_transparency = 0.01 # 0.05 

        closeness = nx.closeness_centrality(G)
        d_clos_unsort  = {}
        for node, cl in sorted(closeness.items(), key = lambda x: x[1], reverse = 0):
            d_clos_unsort [node] = round(cl,4)  
        col_pal = 'viridis'
        d_clos = {key:d_clos_unsort[key] for key in G.nodes()}
        d_nodecol = d_clos
        d_colours = color_nodes_from_dict(G, d_nodecol, palette = col_pal)
        colours = list(d_colours.values())
        node_size = 1.5
        l_feat = list(G.nodes())

        d_degs = dict(G.degree())
        betweens = nx.betweenness_centrality(G)
        d_betw = {}
        for node, be in sorted(betweens.items(), key = lambda x: x[1], reverse = 1):
            d_betw[node] = round(be,4)
        d_degs_sorted = {key:d_degs[key] for key in sorted(d_degs.keys())}
        d_clos_sorted = d_clos
        d_betw_sorted = {key:d_betw[key] for key in sorted(d_betw.keys())}
        feature_dict = dict(zip(d_degs_sorted.keys(), zip(d_degs_sorted.values(), d_clos_sorted.values(), d_betw_sorted.values())))
        feature_dict_sorted = {key:feature_dict[key] for key in G.nodes()}
        DM_imp = pd.DataFrame.from_dict(feature_dict_sorted, orient = 'index', columns = ['degs', 'clos', 'betw'])

        r_scale = 1.2
        umap2D = embed_umap_2D(DM_imp, n_neighbors, spread, min_dist, metric)
        posG = get_posG_2D_norm(G, DM_imp, umap2D, r_scale)      

        nodes = get_trace_nodes_2D(posG, l_feat, colours, node_size) #, node_opac)
        nodes_glow = get_trace_nodes_2D(posG, None, colours, nodesglow_diameter, nodesglow_transparency)
        edges = get_trace_edges_2D(G, posG, edge_colordark, opac = edge_opac)
        data = [nodes_glow,edges, nodes]
        fig = plot2D_app(data)

        return fig , posG, colours

# def portrait2D_func(G):

############################
#
#      PORTRAIT 3D 
#
############################

def portrait3D_local(G):

        n_neighbors = 20 
        spread = 0.9
        min_dist = 0
        metric='cosine'

        edge_width = 0.8
        edge_opac = 0.1
        edge_colordark = '#666666'
        node_size = 1.5
        opacity_nodes = 0.9
        nodesglow_diameter = 20.0
        nodesglow_transparency = 0.01 # 0.05 

        closeness = nx.closeness_centrality(G)
        d_clos_unsort  = {}
        for node, cl in sorted(closeness.items(), key = lambda x: x[1], reverse = 0):
            d_clos_unsort [node] = round(cl,4)  
        col_pal = 'viridis'
        d_clos = {key:d_clos_unsort[key] for key in G.nodes()}
        d_nodecol = d_clos
        d_colours = color_nodes_from_dict(G, d_nodecol, palette = col_pal)
        colours = list(d_colours.values())
        node_size = 1.5
        l_feat = list(G.nodes())

        A = nx.adjacency_matrix(G, nodelist=list(G.nodes()))
        M_adj = A.toarray()
        DM_adj = pd.DataFrame(M_adj)
        DM_adj.index=list(G.nodes())
        DM_adj.columns=list(G.nodes())
        embed3D_local = embed_umap_3D(DM_adj,n_neighbors,spread,min_dist,metric)
        posG_3D_local = get_posG_3D_norm(G,DM_adj,embed3D_local) 
        umap3D_nodes_local = get_trace_nodes_3D(posG_3D_local, l_feat, colours, node_size)
        umap3D_nodes_glow = get_trace_nodes_3D(posG_3D_local, None, colours, nodesglow_diameter, nodesglow_transparency)
        umap3D_edges_local = get_trace_edges_3D(G, posG_3D_local, edge_colordark, opac=edge_opac, linewidth=edge_width) 
        umap3D_data_local = [umap3D_nodes_glow, umap3D_edges_local, umap3D_nodes_local]
        fig3D_local = plot3D_app(umap3D_data_local)
        
        return fig3D_local , posG_3D_local , colours 

def portrait3D_global(G):

        n_neighbors = 20 
        spread = 0.9
        min_dist = 0
        metric='cosine'

        edge_width = 0.8
        edge_opac = 0.2
        edge_colordark = '#666666'
        node_size = 1.5
        opacity_nodes = 0.9
        nodesglow_diameter = 20.0
        nodesglow_transparency = 0.01 # 0.01 

        closeness = nx.closeness_centrality(G)
        d_clos_unsort  = {}
        for node, cl in sorted(closeness.items(), key = lambda x: x[1], reverse = 0):
            d_clos_unsort [node] = round(cl,4)  
        col_pal = 'viridis'
        d_clos = {key:d_clos_unsort[key] for key in G.nodes()}
        d_nodecol = d_clos
        d_colours = color_nodes_from_dict(G, d_nodecol, palette = col_pal)
        colours = list(d_colours.values())
        l_feat = list(G.nodes())

        A = nx.adjacency_matrix(G, nodelist=list(G.nodes()))
        DM_m = pd.DataFrame(rnd_walk_matrix2(A,0.9,1,len(G))).T
        DM_m.index=list(G.nodes())
        DM_m.columns=list(G.nodes())

        embed3D_global = embed_umap_3D(DM_m,n_neighbors,spread,min_dist,metric)
        posG_3D_global = get_posG_3D_norm(G,DM_m,embed3D_global) 
        umap3D_nodes_global = get_trace_nodes_3D(posG_3D_global, l_feat, colours, node_size)
        umap3D_nodes_glow = get_trace_nodes_3D(posG_3D_global, None, colours, nodesglow_diameter, nodesglow_transparency)
        umap3D_edges_global = get_trace_edges_3D(G, posG_3D_global, edge_colordark, opac=edge_opac, linewidth=edge_width) 
        umap3D_data_global = [umap3D_nodes_glow, umap3D_edges_global, umap3D_nodes_global]
        fig3D_global = plot3D_app(umap3D_data_global)   
        
        return fig3D_global ,posG_3D_global, colours

def portrait3D_importance(G):

        n_neighbors = 20 
        spread = 0.9
        min_dist = 0
        metric='cosine'

        edge_width = 0.8
        edge_opac = 0.2
        edge_colordark = '#666666'
        node_size = 1.5
        opacity_nodes = 0.9
        nodesglow_diameter = 20.0
        nodesglow_transparency = 0.01 # 0.01   

        closeness = nx.closeness_centrality(G)
        d_clos_unsort  = {}
        for node, cl in sorted(closeness.items(), key = lambda x: x[1], reverse = 0):
            d_clos_unsort [node] = round(cl,4)  
        col_pal = 'viridis'
        d_clos = {key:d_clos_unsort[key] for key in G.nodes()}
        d_nodecol = d_clos
        d_colours = color_nodes_from_dict(G, d_nodecol, palette = col_pal)
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
        DM_imp = pd.DataFrame.from_dict(feature_dict_sorted, orient = 'index', columns = ['degs', 'clos', 'betw'])

        embed3D_imp = embed_umap_3D(DM_imp,n_neighbors,spread,min_dist,metric)
        posG_3D_imp = get_posG_3D_norm(G,DM_imp,embed3D_imp) 
        umap3D_nodes_imp = get_trace_nodes_3D(posG_3D_imp, l_feat, colours, node_size)
        umap3D_nodes_glow = get_trace_nodes_3D(posG_3D_imp, None, colours, nodesglow_diameter, nodesglow_transparency)
        umap3D_edges_imp = get_trace_edges_3D(G, posG_3D_imp, edge_colordark, opac=edge_opac, linewidth=edge_width) 
        umap3D_data_imp = [umap3D_nodes_glow, umap3D_edges_imp, umap3D_nodes_imp]
        fig3D_imp = plot3D_app(umap3D_data_imp)

        return fig3D_imp ,posG_3D_imp , colours
    
# def portrait3D_func(G):

############################
#
#      TOPOGRAPHIC 
#
############################

def topographic_local(G, z_list):
        
    n_neighbors = 20 
    spread = 0.9
    min_dist = 0
    metric='cosine'

    edge_width = 0.8
    edge_opac = 0.2
    edge_colordark = '#666666'
    node_size = 1.5
    opacity_nodes = 0.9
    #nodesglow_diameter = 20.0
    #nodesglow_transparency = 0.05 # 0.01   

    closeness = nx.closeness_centrality(G)
    d_clos_unsort  = {}
    for node, cl in sorted(closeness.items(), key = lambda x: x[1], reverse = 0):
        d_clos_unsort [node] = round(cl,4)  
    col_pal = 'viridis'
    d_clos = {key:d_clos_unsort[key] for key in G.nodes()}
    d_nodecol = d_clos
    d_colours = color_nodes_from_dict(G, d_nodecol, palette = col_pal)
    colours = list(d_colours.values())
    l_feat = list(G.nodes())

    A = nx.adjacency_matrix(G, nodelist=list(G.nodes()))
    M_adj = A.toarray()
    DM_adj = pd.DataFrame(M_adj)
    DM_adj.index=list(G.nodes())
    DM_adj.columns=list(G.nodes())
    
    z_list_norm = sklearn.preprocessing.minmax_scale(z_list, feature_range=(0, 1.0), axis=0, copy=True)
    r_scale=1.2
    umap2D_local = embed_umap_2D(DM_adj, n_neighbors, spread, min_dist, metric)
    posG_complete_umap_norm_local = get_posG_2D_norm(G, DM_adj, umap2D_local, r_scale)
    posG_land_umap_local = {}
    cc = 0
    for k,v in posG_complete_umap_norm_local.items():
        posG_land_umap_local[k] = (v[0],v[1],z_list_norm[cc])
        cc+=1
                        
    umapland_nodes_local = get_trace_nodes_3D(posG_land_umap_local, l_feat, colours, node_size, opacity_nodes)
    umapland_edges_local = get_trace_edges_3D(G, posG_land_umap_local, edge_colordark, opac=edge_opac, linewidth=edge_width)
    umapland_data_local = [umapland_edges_local, umapland_nodes_local]
    figland_local = plot3D_app(umapland_data_local)
    
    return figland_local ,posG_land_umap_local, colours

def topographic_global(G, z_list):

        n_neighbors = 20 
        spread = 0.9
        min_dist = 0
        metric='cosine'

        edge_width = 0.8
        edge_opac = 0.2
        edge_colordark = '#666666'
        node_size = 1.5
        opacity_nodes = 0.9
        #nodesglow_diameter = 20.0
        #nodesglow_transparency = 0.05 # 0.01 

        closeness = nx.closeness_centrality(G)
        d_clos_unsort  = {}
        for node, cl in sorted(closeness.items(), key = lambda x: x[1], reverse = 0):
            d_clos_unsort [node] = round(cl,4)  
        col_pal = 'viridis'
        d_clos = {key:d_clos_unsort[key] for key in G.nodes()}
        d_nodecol = d_clos
        d_colours = color_nodes_from_dict(G, d_nodecol, palette = col_pal)
        colours = list(d_colours.values())
        node_size = 1.5
        l_feat = list(G.nodes())
        A = nx.adjacency_matrix(G, nodelist=list(G.nodes()))
        DM_m = pd.DataFrame(rnd_walk_matrix2(A,0.9,1,len(G))).T
        DM_m.index=list(G.nodes())
        DM_m.columns=list(G.nodes())
        
        z_list_norm = sklearn.preprocessing.minmax_scale(z_list, feature_range=(0, 1.0), axis=0, copy=True)
        r_scale=1.2
        umap2D_global = embed_umap_2D(DM_m, n_neighbors, spread, min_dist, metric)
        posG_complete_umap_norm_global = get_posG_2D_norm(G, DM_m, umap2D_global, r_scale)
        posG_land_umap_global = {}
        cc = 0
        for k,v in posG_complete_umap_norm_global.items():
            posG_land_umap_global[k] = (v[0],v[1],z_list_norm[cc])
            cc+=1
                        
        umapland_nodes_global = get_trace_nodes_3D(posG_land_umap_global, l_feat, colours, node_size, opacity_nodes)
        umapland_edges_global = get_trace_edges_3D(G, posG_land_umap_global, edge_colordark, opac=edge_opac, linewidth=edge_width)
        umapland_data_global = [umapland_edges_global, umapland_nodes_global]
        figland_global=plot3D_app(umapland_data_global)

        return figland_global ,posG_land_umap_global , colours

def topographic_importance(G, z_list): 

        n_neighbors = 20 
        spread = 0.9
        min_dist = 0
        metric='cosine'

        edge_width = 0.8
        edge_opac = 0.2
        edge_colordark = '#666666'
        node_size = 1.5
        opacity_nodes = 0.9
        #nodesglow_diameter = 20.0
        #nodesglow_transparency = 0.05 # 0.01   

        closeness = nx.closeness_centrality(G)
        d_clos_unsort  = {}
        for node, cl in sorted(closeness.items(), key = lambda x: x[1], reverse = 0):
            d_clos_unsort [node] = round(cl,4)  
        col_pal = 'viridis'
        d_clos = {key:d_clos_unsort[key] for key in G.nodes()}
        d_nodecol = d_clos
        d_colours = color_nodes_from_dict(G, d_nodecol, palette = col_pal)
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
        DM_imp = pd.DataFrame.from_dict(feature_dict_sorted, orient = 'index', columns = ['degs', 'clos', 'betw'])
        
        z_list_norm = sklearn.preprocessing.minmax_scale(z_list, feature_range=(0, 1.0), axis=0, copy=True)
        r_scale=1.2
        umap2D_imp = embed_umap_2D(DM_imp, n_neighbors, spread, min_dist, metric)
        posG_complete_umap_norm_imp = get_posG_2D_norm(G, DM_imp, umap2D_imp, r_scale)
        posG_land_umap_imp = {}
        cc = 0
        for k,v in posG_complete_umap_norm_imp.items():
            posG_land_umap_imp[k] = (v[0],v[1],z_list_norm[cc])
            cc+=1    
        umapland_nodes_imp = get_trace_nodes_3D(posG_land_umap_imp, l_feat, colours, node_size, opacity_nodes)
        umapland_edges_imp = get_trace_edges_3D(G, posG_land_umap_imp, edge_colordark, opac=edge_opac, linewidth=edge_width)
        umapland_data_imp = [umapland_edges_imp, umapland_nodes_imp]
        figland_imp = plot3D_app(umapland_data_imp)

        return figland_imp, posG_land_umap_imp , colours

#def topogrpahic_func(G,z_list):

############################
#
#      GEODESIC 
#
############################

def geodesic_local(G, dict_radius): #, int_restradius):
    
    n_neighbors = 20 
    spread = 0.9
    min_dist = 0
    metric='cosine'

    edge_width = 0.8
    edge_opac = 0.2
    edge_colordark = '#666666'
    node_size = 1.5
    opacity_nodes = 0.9
    #nodesglow_diameter = 20.0
    #nodesglow_transparency = 0.05 # 0.01   

    closeness = nx.closeness_centrality(G)
    d_clos_unsort  = {}
    for node, cl in sorted(closeness.items(), key = lambda x: x[1], reverse = 0):
        d_clos_unsort [node] = round(cl,4)  
    col_pal = 'viridis'
    d_clos = {key:d_clos_unsort[key] for key in G.nodes()}
    d_nodecol = d_clos
    d_colours = color_nodes_from_dict(G, d_nodecol, palette = col_pal)
    colours = list(d_colours.values())
    l_feat = list(G.nodes())

    A = nx.adjacency_matrix(G, nodelist=list(G.nodes()))
    M_adj = A.toarray()
    DM_adj = pd.DataFrame(M_adj)
    DM_adj.index=list(G.nodes())
    DM_adj.columns=list(G.nodes())

    genes = list(G.nodes())
    umap_sphere = embed_umap_sphere(DM_adj, n_neighbors, spread, min_dist, metric)
    #posG_sphere = get_posG_sphere(genes, umap_sphere)
    posG_complete_sphere_norm = get_posG_sphere_norm(G, genes, umap_sphere, dict_radius)#, int_rest_radius)

    umapsphere_nodes = get_trace_nodes_3D(posG_complete_sphere_norm, l_feat, colours, node_size, opacity_nodes)
    #umapsphere_nodes_glow = get_trace_nodes_3D(posG_complete_sphere_norm, l_features, colours, nodesglow_diameter, nodesglow_transparency) 
    umapsphere_edges = get_trace_edges_3D(G, posG_complete_sphere_norm, edge_colordark, opac = edge_opac, linewidth= edge_width)
    umapsphere_data = [umapsphere_edges,umapsphere_nodes]
    figsphere_local = plot3D_app(umapsphere_data)

    return figsphere_local, posG_complete_sphere_norm, colours

def geodesic_global(G,dict_radius):
   
    n_neighbors = 20 
    spread = 0.9
    min_dist = 0
    metric='cosine'

    edge_width = 0.8
    edge_opac = 0.2
    edge_colordark = '#666666'
    node_size = 1.5
    opacity_nodes = 0.9
    #nodesglow_diameter = 20.0
    #nodesglow_transparency = 0.05 # 0.01   

    closeness = nx.closeness_centrality(G)
    d_clos_unsort  = {}
    for node, cl in sorted(closeness.items(), key = lambda x: x[1], reverse = 0):
        d_clos_unsort [node] = round(cl,4)  
    col_pal = 'viridis'
    d_clos = {key:d_clos_unsort[key] for key in G.nodes()}
    d_nodecol = d_clos
    d_colours = color_nodes_from_dict(G, d_nodecol, palette = col_pal)
    colours = list(d_colours.values())
    l_feat = list(G.nodes())

    A = nx.adjacency_matrix(G, nodelist=list(G.nodes()))
    DM_m = pd.DataFrame(rnd_walk_matrix2(A,0.9,1,len(G))).T
    DM_m.index=list(G.nodes())
    DM_m.columns=list(G.nodes())
    
    genes = list(G.nodes())
    umap_sphere = embed_umap_sphere(DM_m, n_neighbors, spread, min_dist, metric)
    #posG_sphere = get_posG_sphere(genes, umap_sphere)
    posG_complete_sphere_norm = get_posG_sphere_norm(G, genes, umap_sphere, dict_radius)#, int_rest_radius)

    umapsphere_nodes = get_trace_nodes_3D(posG_complete_sphere_norm, l_feat, colours, node_size, opacity_nodes)
    #umapsphere_nodes_glow = get_trace_nodes_3D(posG_complete_sphere_norm, l_features, colours, nodesglow_diameter, nodesglow_transparency) 
    umapsphere_edges = get_trace_edges_3D(G, posG_complete_sphere_norm, edge_colordark, opac = edge_opac, linewidth= edge_width)
    umapsphere_data = [umapsphere_edges,umapsphere_nodes]
    figsphere_global = plot3D_app(umapsphere_data)

    return figsphere_global , posG_complete_sphere_norm, colours

def geodesic_importance(G,dict_radius):

    n_neighbors = 20 
    spread = 0.9
    min_dist = 0
    metric='cosine'

    edge_width = 0.8
    edge_opac = 0.2
    edge_colordark = '#666666'
    node_size = 1.5
    opacity_nodes = 0.9
    #nodesglow_diameter = 20.0
    #nodesglow_transparency = 0.05 # 0.01   

    closeness = nx.closeness_centrality(G)
    d_clos_unsort  = {}
    for node, cl in sorted(closeness.items(), key = lambda x: x[1], reverse = 0):
        d_clos_unsort [node] = round(cl,4)  
    col_pal = 'viridis'
    d_clos = {key:d_clos_unsort[key] for key in G.nodes()}
    d_nodecol = d_clos
    d_colours = color_nodes_from_dict(G, d_nodecol, palette = col_pal)
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
    DM_imp = pd.DataFrame.from_dict(feature_dict_sorted, orient = 'index', columns = ['degs', 'clos', 'betw'])

    genes = list(G.nodes())
    umap_sphere = embed_umap_sphere(DM_imp, n_neighbors, spread, min_dist, metric)
    #posG_sphere = get_posG_sphere(genes, umap_sphere)
    posG_complete_sphere_norm = get_posG_sphere_norm(G, genes, umap_sphere, dict_radius)#, int_rest_radius)

    umapsphere_nodes = get_trace_nodes_3D(posG_complete_sphere_norm, l_feat, colours, node_size, opacity_nodes)
    #umapsphere_nodes_glow = get_trace_nodes_3D(posG_complete_sphere_norm, l_features, colours, nodesglow_diameter, nodesglow_transparency) 
    umapsphere_edges = get_trace_edges_3D(G, posG_complete_sphere_norm, edge_colordark, opac = edge_opac, linewidth= edge_width)
    umapsphere_data = [umapsphere_edges,umapsphere_nodes]
    figsphere_imp = plot3D_app(umapsphere_data)

    return figsphere_imp , posG_complete_sphere_norm, colours

#def geodesic_func(G,dict_radius):





def load_graph(organism):
    
    if organism == 'yeast':
    
        data = pickle.load( open( "input/BIOGRID-ORGANISM-Saccharomyces_cerevisiae_S288c-3.5.185.mitab.pickle", "rb" ) )

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


# -------------------------------------------------------------------------------------
# B E N C H M A R K I N G  specific
# -------------------------------------------------------------------------------------

def calc_dist_2D(posG):
    '''
    Validation of Layouts 2D. Calculates distances from layout.
    Return list with distances. 
    '''
    l_x= []
    l_y=[]
    for coords in posG.values():
            l_x.append(coords[0])
            l_y.append(coords[1])
            
    p_dist = []
    for idx,val in enumerate(l_x):
        d_list = []
        for c in range(len(l_x)):
            for yy in l_y:
                d = np.sqrt((l_x[idx]-l_x[c])**2+(l_y[idx]-l_y[c])**2)
            d_list.append(d)
        p_dist.append(d_list)
        
    return p_dist


def calc_dist_3D(posG):
    '''
    Validation of Layouts 3D. Calculates distances from layout.
    Return list with distances. 
    '''
    
    l_x = []
    l_y = []
    l_z = []
    for coords in posG.values():
            l_x.append(coords[0])
            l_y.append(coords[1])
            l_z.append(coords[2])
            
    p_dist = []
    for idx,val in enumerate(l_x):
        d_list = []
        for c in range(len(l_x)):
            d = np.sqrt((l_x[idx]-l_x[c])**2+(l_y[idx]-l_y[c])**2+(l_z[idx]-l_z[c])**2)
        d_list.append(d)
    p_dist.append(d_list)
        
    return p_dist


def get_trace_xy(x,y,trace_name,colour):
    '''
    Get trace 2D.
    Used for distance functions (2D; benchmarking) 
    '''    
    
    trace = pgo.Scatter(
        name = trace_name,
    x = x,
    y = y,
    mode='markers',
    marker=dict(
        size=2,
        color=colour
    ),)
    return trace


def get_trace_xyz(x,y,z,trace_name,colour):
    '''
    Generate 3D trace. 
    Used for distance functions (3D; benchmarking)
    '''    
    
    trace = pgo.Scatter3d(
        x = x,
        y = y,
        z = z,
        mode='markers',
        text=trace_name,
        marker=dict(
            size=2,
            color=colour, 
            line_width=0.5,
            line_color = colour,
        ),)
    return trace


def exec_time(start, end):
    diff_time = end - start
    m, s = divmod(diff_time, 60)
    h, m = divmod(m, 60)
    s,m,h = int(round(s, 3)), int(round(m, 0)), int(round(h, 0))
    print("Execution Time: " + "{0:02d}:{1:02d}:{2:02d}".format(h, m, s))
   
    return m,s


def globallayout_2D(G,n_neighbors, spread, min_dist, metric):
    
    r=0.9
    alpha=1.0

    A = nx.adjacency_matrix(G)
    FM_m_array = rnd_walk_matrix2(A, r, alpha, len(G.nodes()))
    DM_rwr = pd.DataFrame(FM_m_array).T

    umap_rwr_2D = embed_umap_2D(DM_rwr, n_neighbors, spread, min_dist, metric)
    posG_umap_rwr = get_posG_2D(list(G.nodes()), umap_rwr_2D)
    posG_complete_umap_rwr = {key:posG_umap_rwr[key] for key in G.nodes()}

    df_posG = pd.DataFrame(posG_complete_umap_rwr).T
    x = df_posG.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df_posG_norm = pd.DataFrame(x_scaled)

    posG_complete_umap_rwr_norm = dict(zip(list(G.nodes()),zip(df_posG_norm[0].values,df_posG_norm[1].values)))
    
    del DM_rwr
    del df_posG
    
    return posG_complete_umap_rwr_norm


def springlayout_2D(G, itr):
    
    posG_spring2D = nx.spring_layout(G, iterations = itr, dim = 2)

    df_posG = pd.DataFrame(posG_spring2D).T
    x = df_posG.values 
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df_posG_norm = pd.DataFrame(x_scaled)
    
    posG_spring2D_norm = dict(zip(list(G.nodes()),zip(df_posG_norm[0].values,df_posG_norm[1].values)))
    
    del posG_spring2D
    del df_posG
    
    return posG_spring2D_norm


def globallayout_3D(G,n_neighbors, spread, min_dist, metric):
    
    r=0.9
    alpha=1.0
    
    A = nx.adjacency_matrix(G)
    FM_m_array = rnd_walk_matrix2(A, r, alpha, len(G.nodes()))
    DM_rwr = pd.DataFrame(FM_m_array).T

    umap_rwr_3D = embed_umap_3D(DM_rwr, n_neighbors, spread, min_dist, metric)
    posG_umap_rwr = get_posG_3D(list(G.nodes()), umap_rwr_3D)
    posG_complete_umap_rwr = {key:posG_umap_rwr[key] for key in G.nodes()}

    df_posG = pd.DataFrame(posG_complete_umap_rwr).T
    x = df_posG.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df_posG_norm = pd.DataFrame(x_scaled)

    posG_complete_umap_rwr_norm = dict(zip(list(G.nodes()),zip(df_posG_norm[0].values,df_posG_norm[1].values,df_posG_norm[2].values)))
    
    del DM_rwr
    del df_posG
    
    return posG_complete_umap_rwr_norm


def springlayout_3D(G, itr):
    
    posG_spring3D = nx.spring_layout(G, iterations = itr, dim = 3)

    df_posG = pd.DataFrame(posG_spring3D).T
    x = df_posG.values 
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df_posG_norm = pd.DataFrame(x_scaled)
    
    posG_spring3D_norm = dict(zip(list(G.nodes()),zip(df_posG_norm[0].values,df_posG_norm[1].values,df_posG_norm[2].values)))
    
    del posG_spring3D
    del df_posG
    
    return posG_spring3D_norm


def pairwise_layout_distance_2D(G,posG):

    dist_layout = {} 
    for p1,p2 in it.combinations(G.nodes(),2):
        dist_layout[(p1,p2)] = np.sqrt((posG[p1][0]-posG[p2][0])**2 + (posG[p1][1]-posG[p2][1])**2)
        
    return dist_layout


def pairwise_layout_distance_3D(G,posG):

    dist_layout = {} 
    for p1,p2 in it.combinations(G.nodes(),2):
        dist_layout[(p1,p2)] = np.sqrt((posG[p1][0]-posG[p2][0])**2 + (posG[p1][1]-posG[p2][1])**2 + (posG[p1][1]-posG[p2][2])**2)
        
    return dist_layout
 
    
def pairwise_network_distance(G):
    
    dist_network = {}
    for p1,p2 in it.combinations(G.nodes(),2):
        dist_network[(p1,p2)] = nx.shortest_path_length(G,p1,p2, method='dijkstra')

    return dist_network


def pairwise_network_distance_parts(G,list_of_nodes):
    
    dist_network = {}
    for p1,p2 in it.combinations(list_of_nodes,2):
        dist_network[(p1,p2)] = nx.shortest_path_length(G,p1,p2, method='dijkstra')
    
    return dist_network


    
def pearson_corrcoef(dist_network, dist_layout):
    
    d_plot_layout = {}
    for spldist in range(1,int(max(dist_network.values()))+1):
        l_s = []
        for k, v in dist_network.items():
            if v == spldist:
                l_s.append(k)

        l_xy = []
        for nodes in l_s:
            try:
                dxy = dist_layout[nodes]
                l_xy.append(dxy)
            except:
                pass
        d_plot_layout[spldist] = l_xy
    
    print('done layout distances prep')
    l_medians_layout = []
    for k, v in d_plot_layout.items():
        l_medians_layout.append(statistics.median(v))
    
    print('calculate pearson correlation coefficient')
    x = np.array(range(1,int(max(dist_network.values()))+1))
    y = np.array(l_medians_layout)
    r_layout = np.corrcoef(x, y)
    
    return r_layout[0][1]



########################################################################################
#
# C O L O R F U N C T I O N S 
#
########################################################################################


def color_nodes(l_genes, color):

    d_col = {}
    for node in l_genes:
        d_col[str(node)] = color
    
    return d_col


def generate_colorlist_nodes(n):
    '''
    Generate color list based on color count (i.e. nodes to be coloured).
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


def color_nodes_from_dict_unsort(d_to_be_coloured, palette):
    ''' 
    Generate node colors based on dictionary.
    Input: 
    - d_to_be_coloured = dictionary with nodes as keys and values indicating different colors.
    - palette = sns.color palette e.g. 'YlOrRd' 
    
    Return dictionary (randomly sorted) with nodes as keys and assigned color to each node.
    ''' 

    # Colouringg
    colour_groups = set(d_to_be_coloured.values())
    colour_count = len(colour_groups)
    pal = sns.color_palette(palette, colour_count)
    palette = pal.as_hex()

    d_colourgroups = {}
    for n in colour_groups:
        d_colourgroups[n] = [k for k in d_to_be_coloured.keys() if d_to_be_coloured[k] == n]
        
    d_colourgroups_sorted = {key:d_colourgroups[key] for key in sorted(d_colourgroups.keys())}

    d_val_col = {}
    for idx,val in enumerate(d_colourgroups_sorted):
        for ix,v in enumerate(palette):
            if idx == ix:
                d_val_col[val] = v

    d_node_colour = {}
    for y in d_to_be_coloured.items(): # y[0] = node id, y[1] = val
        for x in d_val_col.items(): # x[0] = val, x[1] = (col,col,col)
            if x[0] == y[1]:
                d_node_colour[y[0]]=x[1]
    
    return d_node_colour # colours



def color_nodes_from_dict(G, d_to_be_coloured, palette): 
    ''' 
    Generate node colors based on dictionary.
    Input: 
    - G = Graph 
    - d_to_be_coloured = dictionary with nodes as keys and values indicating different colors.
    - palette = sns.color palette e.g. 'YlOrRd' 
    
    Return dictionary, sorted according to Graph nodes, with nodes as keys and assigned color to each node.
    ''' 
    
    # Colouring
    colour_groups = set(d_to_be_coloured.values())
    colour_count = len(colour_groups)
    pal = sns.color_palette(palette, colour_count)
    palette = pal.as_hex()

    d_colourgroups = {}
    for n in colour_groups:
        d_colourgroups[n] = [k for k in d_to_be_coloured.keys() if d_to_be_coloured[k] == n]
        
    d_colourgroups_sorted = {key:d_colourgroups[key] for key in sorted(d_colourgroups.keys())}

    d_val_col = {}
    for idx,val in enumerate(d_colourgroups_sorted):
        for ix,v in enumerate(palette):
            if idx == ix:
                d_val_col[val] = v

    d_node_colour = {}
    for y in d_to_be_coloured.items(): # y[0] = node id, y[1] = val
        for x in d_val_col.items(): # x[0] = val, x[1] = (col,col,col)
            if x[0] == y[1]:
                d_node_colour[y[0]]=x[1]

    # SORT dict based on G.nodes
    d_node_colour_sorted = dict([(key, d_node_colour[key]) for key in G.nodes()])
    
    return d_node_colour_sorted



def color_nodes_from_list(G, l_nodes, col):
    '''
    Color nodes based on essentiality state.
    Input: 
    - G = graph
    - l_nodes = list of nodes
    - col = string or hex; colour 
    All rest genes will be coloured in grey.
    
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



def color_edges_from_nodelist(G, l_genes, color):
    '''
    Color (highlight) edges from specific node list.
    Input: 
    - G = Graph 
    - l_nodes = list of nodes 
    - color = string; color to hightlight
    
    Return edge list for selected edges. 
    '''
    
    edge_lst = [(u,v) for u,v in G.edges(l_genes) if u in l_genes or v in l_genes]

    d_col_edges = {}
    for e in edge_lst:
        d_col_edges[e]=color

    return d_col_edges


def color_edges_from_nodelist_x(G, l_genes, color):
    '''
    Color (highlight) edges from specific node list exclusively.
    Input: 
    - G = Graph 
    - l_nodes = list of nodes 
    - color = string; color to hightlight
    
    Return edge list for selected edges IF ONE node is IN l_genes. 
    '''
    
    edge_lst = [(u,v)for u,v in G.edges(l_genes) if u in l_genes and v in l_genes]

    d_col_edges = {}
    for e in edge_lst:
        d_col_edges[e]=color

    return d_col_edges



def colours_spectralclustering(G, posG, n_clus, n_comp, pal ='gist_rainbow'):
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

    colours_unsort = color_nodes_from_dict_unsort(d_node_clusterid, pal) #'ocean'
    genes_val = ['#696969']*len(genes_rest)
    colours_rest = dict(zip(genes_rest, genes_val))
    colours_all = {**colours_rest, **colours_unsort}

    d_colours = {key:colours_all[key] for key in G.nodes}
    
    return d_colours



def colours_dbscanclustering(G, DM, posG, epsi, min_sam, pal = 'gist_rainbow', col_rest = '#696969'):
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

    colours_unsort = color_nodes_from_dict_unsort(d_node_clusterid, pal)
    genes_val = [col_rest]*len(genes_rest)
    colours_rest = dict(zip(genes_rest, genes_val))
    colours_all = {**colours_rest, **colours_unsort}

    d_colours_sorted = {key:colours_all[key] for key in G.nodes}
    print('Number of Clusters: ', len(set(clusterid.labels_)))
    
    return d_colours_sorted



def kmeansclustering(posG, n_clus):
    
    df_posG = pd.DataFrame(posG).T 
    kmeans = KMeans(n_clusters=n_clus, random_state=0).fit(df_posG)
    centrs = kmeans.cluster_centers_
    
    return kmeans, centrs



def colours_kmeansclustering(G, DM, kmeans, pal = 'gist_rainbow'):
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
    colours_unsort = color_nodes_from_dict_unsort(d_node_clusterid, pal ) #'prism'
    
    genes_val = ['#696969']*len(genes_rest)
    colours_rest = dict(zip(genes_rest, genes_val))
    colours_all = {**colours_rest, **colours_unsort}
    d_colours_sorted = {key:colours_all[key] for key in G.nodes}
    
    return d_colours_sorted



# -------------------------------------------------------------------------------------
# E S S E N T I A L I T Y   S P E C I F I C  
# -------------------------------------------------------------------------------------


def color_essentiality_nodes(G, essentials, nonessentials, colour1, colour2):
    '''
    Color nodes based on essentiality state.
    Input: 
    - G = graph
    - essentials = list of all essential genes
    - nonessentials = list of all non-essential genes 
    - colour1 = string; to color essential genes
    - colour2 = string; to color non-essential genes 
    All rest genes will be coloured in grey.
    
    Return list of colors for each node in the graph, sorted based on Graph nodes.
    '''

    d_ess = {}
    for node in essentials:
        d_ess[node] = colour1

    d_no_ess = {}
    for node in nonessentials:
        d_no_ess[node] = colour2

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



# -------------------------------------------------------------------------------------
# H U B   S P E C I F I C
# -------------------------------------------------------------------------------------


def identify_hubs(degs, closeness, betweens, cutoff):
    '''
    Identify hubs based on a chosen cutoff.
    Input: 
    - degs/closeness/betweens = each > dictionary with nodes as keys and centrality as values.
    - cutoff: integerfor cut off 
    
    Return nodes to be considered as hubs based on cutoff.
    '''
    
    d_deghubs_cutoff = {}
    for node, de in sorted(degs.items(), key = lambda x: x[1], reverse = 1)[:cutoff]:
        d_deghubs_cutoff[node] = de/max(degs.values())

    d_closhubs_cutoff = {}
    for node, cl in sorted(closeness.items(), key = lambda x: x[1], reverse = 1)[:cutoff]:
        d_closhubs_cutoff[node] = cl

    d_betwhubs_cutoff = {}
    for node, be in sorted(betweens.items(), key = lambda x: x[1], reverse = 1)[:cutoff]:
        d_betwhubs_cutoff[node] = be

    # HUBS SCORE 
    overlap = set(d_deghubs_cutoff.keys()) & set(d_closhubs_cutoff.keys()) & set(d_betwhubs_cutoff.keys())

    d_node_hubs = {}
    for node in overlap:
        d_node_hubs[node] = d_deghubs_cutoff[node]+d_betwhubs_cutoff[node]+d_closhubs_cutoff[node]

    
    return d_node_hubs


   
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
    
    
    return hubs,neighbours



def color_nodes_hubs(G, hubs, neighs, hubs_col_nodes, neigh_col_nodes):
    
    rest_col_nodes = '#d3d3d3' 
    rest_col_edges = '#d3d3d3' 

    colours_hubs = {}
    for i in G.nodes():
        if str(i) in hubs.keys():
            colours_hubs[i] = hubs_col_nodes
        elif str(i) in neighs.keys():
            colours_hubs[i] = neigh_col_nodes
        else: 
            colours_hubs[i] = rest_col_nodes

    hubs_all_sorted = {key:colours_hubs[key] for key in G.nodes()}
    #colours = list(hubs_all_sorted.values())
    
    return hubs_all_sorted 



def color_edges_hubs(G, hubs, hub_col_edges, rest_col_edges):

    d_edge_col_ = color_edges_from_genelist(G, list(hubs.keys()), hub_col_edges)
    d_rest_edges={}
    for e in G.edges():
        if str(e) not in d_edge_col_.keys():
            d_rest_edges[e] = rest_col_edges

    d_all_edges = {**d_edge_col_, **d_rest_edges}
    d_all_edges_sort = {key:d_all_edges[key] for key in G.edges()}
    
    return d_all_edges_sort



def color_nodes_and_neighbors(G, dict_nodes):
    '''
    Generate colors from nodes and also color their neighbors in a lighter color.
    Input: 
    - G = Graph
    - dict_nodes = list of nodes to color 
    Each node will get one color. It's respective neighbor nodes will show in the same but lighter color.
    
    Return colours for each node in the graph, sorted by graph nodes. 
    '''
    
    n = len(set(dict_nodes))
    color = generate_colorlist_nodes(n)

    # LIGHTER COLORS FOR NEIGHBOURING NODES
    factor = 1.7 # the higher the lighter
    color_neigh = []
    for i in color:
        r,g,b = hex_to_rgb(i)
        color_light = adjust_color_lightness(r,g,b,factor)
        color_neigh.append(color_light)
        
    # major coloured nodes
    d_col_major = {}
    for idx,n in enumerate(dict_nodes.keys()):
            d_col_major[n] = color[idx]

    # direct adjacent nodes for major nodes 
    direct_neigh = {}
    for n in d_col_major.keys():
        l = []
        for pair in G.edges():
            if n == pair[0]:
                l.append(pair[1])
                direct_neigh[n] = l
            elif n == pair[1]:
                l.append(pair[0])
                direct_neigh[n] = l

    d_col_neigh = {}
    for node,col in d_col_major.items():
        for idx, node in enumerate(d_col_major.keys()):
            for nd,neigh in direct_neigh.items():
                for n in neigh:
                    if node==nd and n not in d_col_major.keys():
                        d_col_neigh[n]=color_neigh[idx]

    d_col = {**d_col_major,**d_col_neigh}

    # rest nodes
    d_grey = {}
    for i in G.nodes():
        if i not in d_col.keys():
            d_grey[i] = 'dimgrey'

    d_col_all = {**d_col_major, **d_col_neigh, **d_grey}
    d_nodes_colours = {key:d_col_all[key] for key in G.nodes()}
    
    # Node outgoing edges
    edge_lst = []
    for edge in G.edges():
        for e in edge:
            for node in d_col_major.keys():
                if e == node:
                    edge_lst.append(edge)

    # Color edges based on hubs
    d_col_edges = {}
    for e in edge_lst:
        for node,col in d_col_major.items():
            if e[0] == node:
                d_col_edges[e]=col
            elif e[1] == node:
                d_col_edges[e]=col

    d_grey_edges = {}
    for edge in G.edges():
        if edge not in d_col_edges.keys(): 
            d_grey_edges[edge] = 'lightgrey'

    d_edges_all = {**d_col_edges, **d_grey_edges}

    # Sort according to G.edges()
    d_edges_colours = {key:d_edges_all[key] for key in G.edges()}

    return d_nodes_colours, d_edges_colours



# delete eventually (nodes + edges are now merged into one function)
'''def color_majornodes_outgoingedges(G, dict_majorcolor_nodes):

    n = len(set(dict_majorcolor_nodes))
    color = generate_colorlist_nodes(n)

    # LIGHTER COLORS FOR NEIGHBOURING NODES
    factor = 1.7 # the higher the lighter
    color_neigh = []
    for i in color:
        r,g,b = hex_to_rgb(i)
        color_light = adjust_color_lightness(r,g,b,factor)
        color_neigh.append(color_light)
        
    # major coloured nodes
    d_col_major = {}
    for idx,n in enumerate(dict_majorcolor_nodes.keys()):
            d_col_major[n] = color[idx]

    # Node outgoing edges
    edge_lst = []
    for edge in G.edges():
        for e in edge:
            for node in d_col_major.keys():
                if e == node:
                    edge_lst.append(edge)

    # Color edges based on hubs
    d_col_edges = {}
    for e in edge_lst:
        for node,col in d_col_major.items():
            if e[0] == node:
                d_col_edges[e]=col
            elif e[1] == node:
                d_col_edges[e]=col

    d_grey_edges = {}
    for edge in G.edges():
        if edge not in d_col_edges.keys(): 
            d_grey_edges[edge] = 'lightgrey'

    d_edges_all = {**d_col_edges, **d_grey_edges}

    # Sort according to G.edges()
    d_edges_all_sorted = {key:d_edges_all[key] for key in G.edges()}

    edge_color = list(d_edges_all_sorted.values())
    
    return edge_color'''



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


def embed_tsne_2D(Matrix, prplxty, density, l_rate, steps, metric = 'precomputed'):
    '''
    Dimensionality reduction from Matrix using t-SNE.
    Return dict (keys: node IDs, values: x,y).
    ''' 
    
    tsne = TSNE(n_components = 2, random_state = 0, perplexity = prplxty, metric = metric, init='pca',
                     early_exaggeration = density,  learning_rate = l_rate ,n_iter = steps)
    
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


def get_posG_2D_norm(G, DM, embed, r_scalingfactor = 5):
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


def get_trace_edges_2D(G, posG, color_list, opac = 0.2):
    '''
    Get trace of edges for plotting in 2D. 
    Input: 
    - G = Graph
    - posG = dictionary with nodes as keys and coordinates as values.
    - color_list = list of colours obtained from any color function (see above sections).
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
                        line = dict(width = 0.2, color = color_list),
                        opacity = opac
                )
    
    return trace_edges


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
                     early_exaggeration = density,  learning_rate = l_rate, n_iter = n_iter, metric = metric)
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


def get_posG_3D_norm(G, DM, embed):
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


def get_posG_sphere_norm(G, l_genes, sphere_mapper, d_param, radius_rest_genes = 20):
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
        if str(i) in G.nodes():
            genes.append(str(i))

    genes_rest = [] 
    for g in G.nodes():
        if str(g) not in genes:
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


        
########################################################################################
#
# E X P O R T   C O O R D I N A T E S   F U N C T I O N S 
# 
# compatible/for uplad to VRNetzer Platform and Webapp 
#
########################################################################################


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
    df_2D['ID'] = list(posG.keys())

    cols = df_2D.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df_2D_final = df_2D[cols]
    
    return df_2D_final
    

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
    df_3D['ID'] = list(posG.keys())

    cols = df_3D.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df_3D_final = df_3D[cols]
    
    return df_3D_final



########################################################################################
# 
# G E N E   I D / S Y M B O L   F U N C T I O N S  
#
########################################################################################


# GENE entrezID <-> Gene Symbol 

def genent2sym():
    '''
    Return two dictionaries.
    First with gene entrezid > symbol. Second with symbol > entrezid. 
    '''
    
    db = mysql.connect("menchelabdb.int.cemm.at","readonly","ra4Roh7ohdee","GenesGO")    

    # prepare a cursor object using cursor() method
    cursor = db.cursor()

    sql = """   SELECT
                    Approved_Symbol,
                    Entrez_Gene_ID_NCBI 
                FROM GenesGO.hgnc_complete
                WHERE Entrez_Gene_ID_NCBI != ''
          """ 

    cursor.execute(sql)
    data = cursor.fetchall()    
#     try: 
#         # execute SQL query using execute() method.
#         cursor.execute(sql)
#         data = cursor.fetchall()
#     except:
#         print('SQL error')
    db.close()

#     t0 = time.time()
    d_sym_ent = {}
    d_ent_sym = {}

    for x in data:
        sym = x[0]
        ent = x[1]
        d_sym_ent[sym] = ent
        d_ent_sym[ent] = sym
#     print(time.time()-t0)
    
    return d_ent_sym, d_sym_ent



# Gene entrezID <-> Gene Symbol 

def convert_symbol_to_entrez(gene_list,name_species):   #name_species must be the official entrez name in string format
    '''
    Get gene list and name of species and
    Return a dict of Gene Symbol and EntrezID
    '''
    
    sym_to_entrez_dict={}    #create a dictionary symbol to entrez
    for gene in gene_list:
        #retrieve gene ID
        handle = Entrez.esearch(db="gene", term=name_species+ "[Orgn] AND " + gene + "[Gene]")
        record = Entrez.read(handle)

        if len(record["IdList"]) > 0:
            sym_to_entrez_dict[gene]=record["IdList"][0]
        else:
            pass
    return sym_to_entrez_dict


########################################################################################
#
# N O T  I N  U S E  ??? 
#
########################################################################################

def color_edges_from_list(G, genelist, col):
    '''
    Color edges based on essentiality state.
    Input: 
    - G = graph
    - genelist = list of all genes to take into consideration 
    - colour = string; to color gene edges 
    All rest edges will be coloured in grey.
    
    Return list of colors for each edge, sorted based on Graph edges.
    '''
    
    # EDGES ------------------------------
    edge_lst = []
    for edge in G.edges():
        for e in edge:
            for node in d_genes.keys():
                if e == node:
                    edge_lst.append(edge)

    d_col_edges = {}
    for node,col in d_genes.items():
            if e[0] == node:
                d_col_edges[e]= col
            elif e[1] == node:
                d_col_edges[e]= col

    d_grey_edges = {}
    for edge in G.edges():
        if edge not in d_col_edges.keys(): 
            d_grey_edges[edge] = '#d3d3d3'

    d_edges_all = {**d_col_edges, **d_grey_edges}

    # Sort according to G.edges()
    d_edges_all_sorted = {key:d_edges_all[key] for key in G.edges()}

    edge_color = list(d_edges_all_sorted.values())
    
    return edge_color


def color_diseasecategory(G, d_names_do, d_do_genes, disease_category, colour):
    
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
    
    # assign colours to disease cat.(colour1) and other nodes(grey)
    d_col = {}
    for node in set_disease_genes:
        d_col[node] = colour
    
    d_rest = {}
    for i in G.nodes():
        if i not in d_col.keys():
            d_rest[i] = '#303030' # 'dimgrey'
        
    d_allnodes_col = {**d_col, **d_rest}
    d_allnodes_col_sorted = {key:d_allnodes_col[key] for key in G.nodes()}
    
    colours = list(d_allnodes_col_sorted.values())
    
    return colours



def only_numerics(seq):
    seq_type= type(seq)
    return seq_type().join(filter(seq_type.isdigit, seq))


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
    
    d_col_major = {}
    for n in l_nodes:
            d_col_major[n] = color_main

    edge_lst = []
    for edge in G.edges():
        for e in edge:
            if e in d_col_major.keys():
                #if e == node:
                edge_lst.append(edge)

    d_col_edges = {}
    for e in edge_lst:
        for node,col in d_col_major.items():
            if e[0] == node:
                d_col_edges[e]=col
            elif e[1] == node:
                d_col_edges[e]=col

    d_grey_edges = {}
    for edge in G.edges():
        if edge not in d_col_edges.keys(): 
            d_grey_edges[edge] = color_rest# '#d3d3d3'

    d_edges_all = {**d_col_edges, **d_grey_edges}

    # Sort according to G.edges()
    #d_edges_all_sorted = {key:d_edges_all[key] for key in G.edges()}

    #edge_color = list(d_edges_all_sorted.values())
    
    return  d_edges_all