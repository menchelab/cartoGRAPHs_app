from networkx.readwrite.edgelist import parse_edgelist
from pkg_resources import get_default_cache
from plotly.missing_ipywidgets import FigureWidget
from app_main import *
import csv 
import dash_bootstrap_components as dbc
from plotly.io import to_image
import base64
from flask import Flask
#from flask import send_from_directory
from flask import send_file
#from flask import request
#from base64 import b64encode

import dash_table
from dash.exceptions import PreventUpdate
#from plotly.io import write_image
#from flask import Flask, send_from_directory
from urllib.parse import quote as urlquote
import urllib


# Initialise the app
myServer = Flask(__name__)
app = dash.Dash(server=myServer)
                #prevent_initial_callbacks=True) #,suppress_callback_exceptions=True) 



##################################################################################
##################################################################################
#
#                          P R E P   F O R   A P P                   
#
##################################################################################
##################################################################################

############################
#
# Graph parse - function 
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


############################
#
# PPI precalculated 
#
############################

G_ppi = nx.read_edgelist('input/ppi_elist.txt')
port3d_ppi = 'input/3D_global_layout_human.csv'
topo_ppi = 'input/topographic_global_layout_human.csv'
geo_ppi = 'input/geodesic_global_layout_human.csv'

def import_vrnetzer_csv(G,file):

    edge_width = 0.8
    edge_opac = 0.05
    edge_colordark = '#666666'
    node_size = 1.0
    #nodesglow_diameter = 20.0
    #nodesglow_transparency = 0.05 # 0.01 

    df = pd.read_csv(file, header=None)
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
        edge_opac = 0.05
        edge_colordark = '#666666'
        node_edge_col = '#696969'
        node_size = 1.0
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
        M_adj = A.toarray()
        DM_adj = pd.DataFrame(M_adj)
        DM_adj.index=list(G.nodes())
        DM_adj.columns=list(G.nodes())

        r_scale = 1.2
        umap2D = embed_umap_2D(DM_adj, n_neighbors, spread, min_dist, metric)
        posG = get_posG_2D_norm(G, DM_adj, umap2D, r_scale)      

        nodes = get_trace_nodes_2D(posG, l_feat, colours, node_size, linewidth=0.4)
        edges = get_trace_edges_2D(G, posG, edge_colordark, opac = edge_opac)
        data = [edges, nodes]
        fig = plot2D_app(data)

        return fig , posG, colours

def portrait2D_global(G):

        n_neighbors = 20 
        spread = 0.9
        min_dist = 0
        metric='cosine'

        edge_width = 0.8
        edge_opac = 0.05
        edge_colordark = '#666666'
        node_edge_col = '#696969'
        node_size = 1.0
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
  
        r_scale = 1.2
        umap2D = embed_umap_2D(DM_m, n_neighbors, spread, min_dist, metric)
        posG = get_posG_2D_norm(G, DM_m, umap2D, r_scale)      

        nodes = get_trace_nodes_2D(posG, l_feat, colours, node_size, linewidth=0.4)
        edges = get_trace_edges_2D(G, posG, edge_colordark, opac = edge_opac)
        data = [edges, nodes]
        fig = plot2D_app(data)

        return fig , posG, colours

def portrait2D_importance(G):

        n_neighbors = 20 
        spread = 0.9
        min_dist = 0
        metric='cosine'

        edge_width = 0.8
        edge_opac = 0.05
        edge_colordark = '#666666'
        node_edge_col = '#696969'
        node_size = 1.0
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

        r_scale = 1.2
        umap2D = embed_umap_2D(DM_imp, n_neighbors, spread, min_dist, metric)
        posG = get_posG_2D_norm(G, DM_imp, umap2D, r_scale)      

        nodes = get_trace_nodes_2D(posG, l_feat, colours, node_size, linewidth=0.4)
        edges = get_trace_edges_2D(G, posG, edge_colordark, opac = edge_opac)
        data = [edges, nodes]
        fig = plot2D_app(data)

        return fig , posG, colours

# def portrait2D_func(G):

############################
#
#      PORTRAIT 3D 
#
############################

def portrait3D_local_old(G):

        n_neighbors = 20 
        spread = 0.9
        min_dist = 0
        metric='cosine'

        edge_width = 0.8
        edge_opac = 0.05
        edge_colordark = '#666666'
        node_size = 1.0
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
        M_adj = A.toarray()
        DM_adj = pd.DataFrame(M_adj)
        DM_adj.index=list(G.nodes())
        DM_adj.columns=list(G.nodes())
        embed3D_local = embed_umap_3D(DM_adj,n_neighbors,spread,min_dist,metric)
        posG_3D_local = get_posG_3D(list(G.nodes()),embed3D_local) 
        umap3D_nodes_local = get_trace_nodes_3D(posG_3D_local, l_feat, colours, node_size)
        umap3D_edges_local = get_trace_edges_3D(G, posG_3D_local, edge_colordark, opac=edge_opac, linewidth=edge_width) 
        umap3D_data_local = [umap3D_edges_local, umap3D_nodes_local]
        fig3D_local = plot3D_app(umap3D_data_local)
        
        return fig3D_local

def portrait3D_local(G):

        n_neighbors = 20 
        spread = 0.9
        min_dist = 0
        metric='cosine'

        edge_width = 0.8
        edge_opac = 0.05
        edge_colordark = '#666666'
        node_size = 1.0
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
        M_adj = A.toarray()
        DM_adj = pd.DataFrame(M_adj)
        DM_adj.index=list(G.nodes())
        DM_adj.columns=list(G.nodes())
        embed3D_local = embed_umap_3D(DM_adj,n_neighbors,spread,min_dist,metric)
        posG_3D_local = get_posG_3D_norm(G,DM_adj,embed3D_local) 
        umap3D_nodes_local = get_trace_nodes_3D(posG_3D_local, l_feat, colours, node_size)
        umap3D_edges_local = get_trace_edges_3D(G, posG_3D_local, edge_colordark, opac=edge_opac, linewidth=edge_width) 
        umap3D_data_local = [umap3D_edges_local, umap3D_nodes_local]
        fig3D_local = plot3D_app(umap3D_data_local)
        
        return fig3D_local , posG_3D_local , colours 

def portrait3D_global(G):

        n_neighbors = 20 
        spread = 0.9
        min_dist = 0
        metric='cosine'

        edge_width = 0.8
        edge_opac = 0.05
        edge_colordark = '#666666'
        node_size = 1.0
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

        embed3D_global = embed_umap_3D(DM_m,n_neighbors,spread,min_dist,metric)
        posG_3D_global = get_posG_3D_norm(G,DM_m,embed3D_global) 
        umap3D_nodes_global = get_trace_nodes_3D(posG_3D_global, l_feat, colours, node_size)
        umap3D_edges_global = get_trace_edges_3D(G, posG_3D_global, edge_colordark, opac=edge_opac, linewidth=edge_width) 
        umap3D_data_global = [umap3D_edges_global, umap3D_nodes_global]
        fig3D_global = plot3D_app(umap3D_data_global)   
        
        return fig3D_global ,posG_3D_global, colours

def portrait3D_importance(G):

        n_neighbors = 20 
        spread = 0.9
        min_dist = 0
        metric='cosine'

        edge_width = 0.8
        edge_opac = 0.05
        edge_colordark = '#666666'
        node_size = 1.0
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
        umap3D_edges_imp = get_trace_edges_3D(G, posG_3D_imp, edge_colordark, opac=edge_opac, linewidth=edge_width) 
        umap3D_data_imp = [umap3D_edges_imp, umap3D_nodes_imp]
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
    edge_opac = 0.05
    edge_colordark = '#666666'
    node_size = 1.0
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
        edge_opac = 0.05
        edge_colordark = '#666666'
        node_size = 1.0
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
        edge_opac = 0.05
        edge_colordark = '#666666'
        node_size = 1.0
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
    edge_opac = 0.05
    edge_colordark = '#666666'
    node_size = 1.0
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
    edge_opac = 0.05
    edge_colordark = '#666666'
    node_size = 1.0
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
    edge_opac = 0.05
    edge_colordark = '#666666'
    node_size = 1.0
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



##################################################################################
##################################################################################
#
#                          A P P     S T A R T S     H E R E                   
#
##################################################################################
##################################################################################


app.layout = html.Div(
        id='app__banner', 
        children=[
                ######################################
                #
                #           BANNER / LOGO
                #
                ######################################
                html.Div(className="app__banner",
                children=[ 
                    html.Img(src='assets/cartoGraphs_logo_long2.png',style={'height':'70px'}),
                    ],
                ),
    
                ######################################
                #
                # USER INTERFACE / INTERACTIVE PART
                #
                ######################################
                html.Div(className = 'three columns', 
                    children = [

                #########################################
                #
                #  INPUT + UPLOADS
                #
                #########################################

                        #----------------------------------------
                        # Input Graph 
                        #----------------------------------------
                        html.H6('INPUT DATA'),
                        html.P('Upload an edge list or choose a model network.'),
                        dcc.Upload(
                                id='upload-data',
                                children=html.Div([
                                    html.A('Upload an edgelist here.')
                                ]),
                                style={
                                    'width': '99%',
                                    'height': '30px',
                                    'lineHeight': '30px',
                                    'borderWidth': '1.2px',
                                    'borderStyle': 'dashed',
                                    'borderRadius': '5px',
                                    'textAlign': 'center',
                                    'margin-left': '0px', 
                                    'margin-right': '0px', 
                                    'margin-top': '13.5px',
                                    'margin-bottom': '0px', 
                                    'font-size':'12px',
                                    'borderColor':'white',
                                },
                                multiple=False # Allow multiple files to be uploaded
                            ),
                        #html.Div(id='output-data-upload'),

                        #----------------------------------------
                        # Choose Model Network - Button
                        #----------------------------------------
                        html.Div(children=[
                            html.Button('MODEL NETWORK', id='button-network-type', n_clicks=0 ,
                            style={'text-align': 'center','width': '100%','margin-top': '5px'}),
                        ]),

                        #html.Br(),

                        #----------------------------------------
                        # LAYOUTS (local, global, imp, func)
                        #----------------------------------------
                        html.H6('NETWORK LAYOUT'),
                        html.P('Choose one of the listed layouts.'),
                        html.Div(children=[
                            dcc.Dropdown(className='app__dropdown',
                                id='dropdown-layout-type',
                                options=[
                                    {'label': 'local', 'value': 'local'},
                                    {'label': 'global', 'value': 'global'},
                                    {'label': 'importance', 'value': 'importance'},
                                    #{'label': 'functional', 'value': 'functional'},
                                ],
                                placeholder="Select a Network Layout.", 
                                )
                            ]),
                
                        #----------------------------------------
                        # MAP CATEGORY (2D,3D,topo,geo)
                        #----------------------------------------
                        html.H6('NETWORK MAP CATEGORY'),
                        html.P('Choose one of the listed map categories.'),
                        html.Div(children=[
                            dcc.Dropdown(className='app__dropdown',
                                id='dropdown-map-type',
                                options=[
                                    {'label': '2D Portrait', 'value': 'fig2D'},
                                    {'label': '3D Portrait', 'value': 'fig3D'},
                                    {'label': 'Topographic Map', 'value': 'figland'},
                                    {'label': 'Geodesic Map', 'value': 'figsphere'},
                                ],
                                placeholder="Select a Layout Map.", 
                                )
                        ]),
                        
                        #----------------------------------------
                        # UPDATE NETWORK BUTTON
                        #----------------------------------------
                        html.Button('DRAW LAYOUT', id='button-graph-update', n_clicks=0 ,
                            style={'text-align': 'center','width': '100%','margin-top': '5px'}),

                        html.Br(),
                        html.Br(),

                    ]),

                        
                #html.Br(),
                #html.Br(),


                #########################################
                #
                #           GRAPH FIGURE 
                #
                #########################################
                html.Div(className = 'seven columns', 
                    children = [

                    dcc.Loading(
                        id="loading-2",
                        type="circle",
                        style={'display':'center'},
                        children=[
                        dcc.Graph(
                                id='layout-graph-figure',
                                style = {'display':'block', 'width':'100%','height':'80vh'}
                                ),   
                        dash_table.DataTable(
                            id='layout-graph-table',
                                )  
                            ]),
                    
                ]),


                ######################################
                #
                #           DOWNLOADS  
                #
                ######################################
                html.Div(className = 'three columns', #style={'margin-left':'10px'},
                    children = [ 
                        
                        #----------------------------------------
                        # DOWNLOAD Layouts
                        #----------------------------------------
                        html.H6('DOWNLOADS'),
                        html.P('Download Layouts here.'),
                        

                        html.A(
                                id="download-figure", 
                                href="", 
                                children=[html.Button('FIGURE | html', id='button-figure', n_clicks=0,
                                   style={'text-align': 'center', 'width': '100%', 'margin-top': '5px'}),
                        ],
                        ),


                        html.A(
                                id="download-csv", 
                                href="", 
                                children=[html.Button('TABLE | csv', id='button-csv', n_clicks=0 ,
                                   style={'text-align': 'center', 'width': '100%', 'margin-top': '5px'}),
                                   ], 
                        ),


                        #html.Button('3Dprint | OBJ', id='button-obj', n_clicks=0 ,
                        #            style={'text-align': 'center', 'width': '100%', 'margin-top': '5px'}),


                        html.Br(),
                        html.Br(),

                        #----------------------------------------
                        # Paper Figures
                        #----------------------------------------
                        #html.Img(src='assets/netimage4.png',
                        #        style={'height':'220px','width':'100%'
                        #        }),
                        html.H6('EXPLORE THE HUMAN INTERACTOME'),
                        html.P('View Layouts of the Human Protein-Protein Interaction Network.'),

                        #html.Button('2D PORTRAIT', id='button-ppi-2d', n_clicks=0 ,   
                        #            style={'text-align': 'center', 
                        #            'width': '100%', 'margin-top': '5px', #'margin-right':'2px',#'display':'inline-block',
                        #           }),
                        #dcc.Download(id='download-ppi2d'),
                        

                        html.Button('3D PORTRAIT', id='button-ppi-3d', n_clicks=0 ,   
                                    style={'text-align': 'center', 
                                    'width': '100%', 'margin-top': '5px', #'margin-right':'2px',#'display':'inline-block',
                                    }),
                        dcc.Download(id='download-ppi3d'),
                    
                        
                        html.Button('TOPOGRAPHIC MAP', id='button-ppi-topo', n_clicks=0 ,
                                    style={'text-align': 'center', 
                                    'width': '100%', 'margin-top': '5px', #'display':'inline-block',
                                    }),
                        dcc.Download(id='download-ppitopo'),

                        
                        html.Button('GEODESIC MAP', id='button-ppi-geo', n_clicks=0 ,
                                    style={'text-align': 'center', 
                                    'width': '100%', 'margin-top': '5px', #'margin-left':'2px', #'display':'inline-block',
                                    }),
                        dcc.Download(id='download-ppigeo'),

                    ]), 

                html.Div(className = 'footer',
                    children=[
                        html.P('This visualization app is currently under construction. Feel free to get in touch for bug reports, comments, suggestions via Github/menchelab/cartoGRAPHs'),
                    ]),
            ])
        #])




#########################################
#
#          CALL BACKS + related                
#
#########################################

#----------------------------------------
# DOWNLOAD CSV
# #----------------------------------------
@app.callback(
    Output('download-csv', 'href'),
    [Input('button-csv', 'n_clicks')], 
    [Input('layout-graph-table','data')]
    )

def get_table(n_clicks,table):
    #if n_clicks:
            for i in table:
                df = pd.DataFrame(i)
            csv_string = df.to_csv(index=False, header=False, encoding='utf-8')
            csv_string = "data:text/csv;charset=utf-8," + urlquote(csv_string)
            return csv_string

@myServer.route("/download/urlToDownload")
def download_table():
    return dcc.send_dataframe('output/download_figure.csv',
                     mimetype='text:csv',
                     attachment_filename='downloadFile.csv',
                     as_attachment=True
                     )

#------------------------------------
# DOWNLOAD FIGURE 
#------------------------------------
# @app.callback(Output('download-figure', 'href'),
#             [Input('button-figure', 'n_clicks'),
#               #Input('layout-graph-figure','figure')
#               ],
# )
# def make_image(n_clicks):
#     if n_clicks:
#         file_path = os.path.join('output','.html')
#         return dcc.send_file(file_path)



#------------------------------------
# PPI / Figures Manuscript
#------------------------------------
#ppi_portrait3d = import_vrnetzer_csv(G_ppi,port3d_ppi)
#ppi_topographic = import_vrnetzer_csv(G_ppi, topo_ppi)
#ppi_geodesic = import_vrnetzer_csv(G_ppi, geo_ppi)





#----------------------------------------
# Network Layout + Map
#----------------------------------------
@app.callback(

            [Output('layout-graph-figure', 'figure'),
             Output('layout-graph-table', 'data')],

            # button for starting graph
              [Input('button-graph-update','n_clicks')],

            # button for upload 
              Input('upload-data', 'contents'),

            # network input 
              [Input('button-network-type', 'n_clicks')],

            # state of upload
              Input('upload-data', 'filename'),

            # states of layout and map 
              [State('dropdown-layout-type','value')],
              [State('dropdown-map-type', 'value')],
              )

def update_graph(buttonclicks, #'button-graph-update'
                inputcontent, #'upload-data'
                modelclicks, #'button-network-type'
                inputfile, #'upload-data'
                layoutvalue, 
                mapvalue):
            
            #---------------------------------------
            # very start of app 
            #---------------------------------------
            if buttonclicks == 0:

                G = nx.read_edgelist('input/GPPI_sub_1000.txt')

                fig3D_start,posG,colours = portrait3D_local(G)
                namespace='local3d'
                df_vrnetzer = export_to_csv3D_app(namespace,posG,colours)
                dict_vrnetzer = [df_vrnetzer.to_dict()]

                return fig3D_start, dict_vrnetzer
               
            #---------------------------------------
            # Model Graph
            #---------------------------------------
            if modelclicks == 0:
                G = nx.read_edgelist('input/GPPI_sub_1000.txt')
                    
            #---------------------------------------
            # Upload / Input Graph
            #---------------------------------------
            elif inputfile:
                G = parse_Graph(inputcontent,inputfile)        
                    
            #---------------------------------------
            # Model Graph
            #---------------------------------------
            elif modelclicks:
                G = nx.read_edgelist('input/GPPI_sub_1000.txt')
                    


            if buttonclicks:
                #---------------------------------------
                # Toggling between layouts
                #---------------------------------------
                ##################
                #
                #  2 d PORTRAIT
                #
                ##################
                if mapvalue == 'fig2D':
                    if layoutvalue == 'local': 
                        fig2D_local,posG,colours = portrait2D_local(G) 

                        namespace='local2d'
                        df_vrnetzer = export_to_csv2D_app(namespace,posG,colours)
                        dict_vrnetzer = [df_vrnetzer.to_dict()]  

                        return fig2D_local, dict_vrnetzer

                    elif layoutvalue == 'global':
                        fig2D_global,posG,colours = portrait2D_global(G) 

                        namespace='global2d'
                        df_vrnetzer = export_to_csv2D_app(namespace,posG,colours)
                        dict_vrnetzer = [df_vrnetzer.to_dict()]  

                        return fig2D_global,dict_vrnetzer

                    elif layoutvalue == 'importance':  
                        fig2D_imp,posG,colours = portrait2D_importance(G) 

                        namespace='imp2d'
                        df_vrnetzer = export_to_csv2D_app(namespace,posG,colours)
                        dict_vrnetzer = [df_vrnetzer.to_dict()]  

                        return fig2D_imp, dict_vrnetzer

                    # if layoutvalue == 'func':



                ##################
                #
                #  3 D PORTRAIT
                #
                ##################
                elif mapvalue == 'fig3D':

                    if layoutvalue == 'local':
                        fig3D_local,posG,colours = portrait3D_local(G)

                        namespace='local3d'
                        df_vrnetzer = export_to_csv3D_app(namespace,posG,colours)
                        dict_vrnetzer = [df_vrnetzer.to_dict()]  

                        return fig3D_local, dict_vrnetzer

                    elif layoutvalue == 'global':
                        fig3D_global,posG,colours = portrait3D_global(G)

                        namespace='global3d'
                        df_vrnetzer = export_to_csv3D_app(namespace,posG,colours)
                        dict_vrnetzer = [df_vrnetzer.to_dict()]
                        
                        return fig3D_global, dict_vrnetzer
                
                    elif layoutvalue == 'importance':
                        fig3D_imp, posG, colours = portrait3D_importance(G)

                        namespace='imp3d'
                        df_vrnetzer = export_to_csv3D_app(namespace,posG,colours)
                        dict_vrnetzer = [df_vrnetzer.to_dict()]

                        return fig3D_imp, dict_vrnetzer


                    #elif layoutvalue == 'functional':
                    #    fig3D_func = portrait3D_func(G)
                    #    return html.Div(id='layout-graph',children= [
                    #                            dcc.Graph(
                    #                                    config={'displayModeBar':False},
                    #                                    style={'position':'relative','height': '80vh', 'width':'100%'},
                    #                                    figure=fig3D_func
                    #                                   ),
                    #                              ])

                
                ##################
                #
                #  TOPOGRAPHIC
                #
                ##################
                elif mapvalue == 'figland':
                    deg = dict(G.degree())
                    z_list = list(deg.values()) # U P L O A D L I S T  with values if length G.nodes !!! 
                    
                    if layoutvalue == 'local':
                        figland_local,posG,colours = topographic_local(G,z_list)

                        namespace='localtopo'
                        df_vrnetzer = export_to_csv3D_app(namespace,posG,colours)
                        dict_vrnetzer = [df_vrnetzer.to_dict()]

                        return figland_local , dict_vrnetzer

                    elif layoutvalue == 'global':
                        figland_global,posG,colours = topographic_global(G,z_list) 

                        namespace='globaltopo'
                        df_vrnetzer = export_to_csv3D_app(namespace,posG,colours)
                        dict_vrnetzer = [df_vrnetzer.to_dict()]  

                        return figland_global,dict_vrnetzer 
                    
                    elif layoutvalue == 'importance':
                        
                        closeness = nx.closeness_centrality(G)
                        d_clos_unsort  = {}
                        for node, cl in sorted(closeness.items(), key = lambda x: x[1], reverse = 0):
                            d_clos_unsort [node] = round(cl,4)  
                        d_clos = {key:d_clos_unsort[key] for key in G.nodes()}
                        z_list = list(d_clos.values())
                        
                        figland_imp,posG,colours = topographic_importance(G, z_list)

                        namespace='imptopo'
                        df_vrnetzer = export_to_csv3D_app(namespace,posG,colours)
                        dict_vrnetzer = [df_vrnetzer.to_dict()]  

                        return figland_imp,dict_vrnetzer 


                    #elif layoutvalue == 'functional':
                    #    figland_func = topographic_func(G z_list)
                    #    return html.Div(id='layout-graph',children= [
                    #                                    dcc.Graph(
                    #                                            config={'displayModeBar':False},
                    #                                            style={'position':'relative','height': '80vh', 'width':'100%'},
                    #                                            figure=figland_func
                    #                                            ),
                    #                                        ])

                
                ##################
                #
                #  GEODESIC 
                #
                ##################
                elif mapvalue == 'figsphere':
                    radius = dict(G.degree()) # U P L O A D L I S T  with values if length G.nodes !!! 

                    if layoutvalue == 'local':
                        figsphere_local,posG,colours = geodesic_local(G,radius)
                        
                        namespace='localgeo'
                        df_vrnetzer = export_to_csv3D_app(namespace,posG,colours)
                        dict_vrnetzer = [df_vrnetzer.to_dict()]  

                        return figsphere_local,dict_vrnetzer

                    elif layoutvalue == 'global':  
                        figsphere_global,posG,colours = geodesic_global(G,radius) 

                        namespace='localgeo'
                        df_vrnetzer = export_to_csv3D_app(namespace,posG,colours)
                        dict_vrnetzer = [df_vrnetzer.to_dict()]  

                        return figsphere_global,dict_vrnetzer
                    
                    elif layoutvalue == 'importance':
                        figsphere_imp,posG,colours = geodesic_importance(G,radius)
                        
                        namespace='localgeo'
                        df_vrnetzer = export_to_csv3D_app(namespace,posG,colours)
                        dict_vrnetzer = [df_vrnetzer.to_dict()]  

                        return figsphere_imp , dict_vrnetzer

                    #elif layoutvalue == 'functional':
                    #    figsphere_func = geodesic_func(G,radius)
                    #    return html.Div(id='layout-graph',children= [
                    #                                    dcc.Graph(
                    #                                            config={'displayModeBar':False},
                    #                                            style={'position':'relative','height': '80vh', 'width':'100%'},
                    #                                            figure=figsphere_func
                    #                                            ),
                    #                                        ])




server = app.server
if __name__ == '__main__':
    app.run_server(debug=True)





                        #------------------------- 
                        # functional data matrix   
                        # >> use something like for matrix input : 
                        # A = pd.read_csv(input, sep='\t', index_col=0, header=0)
                        # Ai = A.applymap(inverse)
                        # G = nx.from_pandas_adjacency(Ai)
                        #from here: https://community.plotly.com/t/converting-networkx-graph-object-into-cytoscape-format/23224/6
                        #-------------------------
                        #dcc.Upload(
                        #        id='upload-matrix',
                        #        children=html.Div([
                        #            html.A('Upload matrix here.')
                        #        ]),
                        #        style={
                        #            'width': '100%',
                        #            'height': '32px',
                        #            'lineHeight': '32px',
                        #            'borderWidth': '1.2px',
                        #            'borderStyle': 'dashed',
                        #           'borderRadius': '5px',
                        #            'textAlign': 'center',
                        #            'margin': '0px', 
                        #            'font-size':'12px',
                        #        },
                        #        multiple=False# Allow multiple files to be uploaded
                        #   ),
                        #html.Br(),