from networkx.readwrite.edgelist import parse_edgelist
from app_main import *
import csv 
import dash_bootstrap_components as dbc
from plotly.io import to_image
import base64
from flask import Flask
from flask import send_from_directory
from flask import send_file
from flask import request


# Initialise the app
myServer = Flask(__name__)
app = dash.Dash(server=myServer)#prevent_initial_callbacks=True)


##################################################################################
##################################################################################
#
#                          P R E P   F O R   A P P                   
#
##################################################################################
##################################################################################


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

        return fig 

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

        return fig 

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

        return fig 

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
        posG_3D_global = get_posG_3D(list(G.nodes()),embed3D_global) 
        umap3D_nodes_global = get_trace_nodes_3D(posG_3D_global, l_feat, colours, node_size)
        umap3D_edges_global = get_trace_edges_3D(G, posG_3D_global, edge_colordark, opac=edge_opac, linewidth=edge_width) 
        umap3D_data_global = [umap3D_edges_global, umap3D_nodes_global]
        fig3D_global = plot3D_app(umap3D_data_global)   
        
        return fig3D_global 

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
        posG_3D_imp = get_posG_3D(list(G.nodes()),embed3D_imp) 
        umap3D_nodes_imp = get_trace_nodes_3D(posG_3D_imp, l_feat, colours, node_size)
        umap3D_edges_imp = get_trace_edges_3D(G, posG_3D_imp, edge_colordark, opac=edge_opac, linewidth=edge_width) 
        umap3D_data_imp = [umap3D_edges_imp, umap3D_nodes_imp]
        fig3D_imp = plot3D_app(umap3D_data_imp)

        return fig3D_imp 
    
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
    
    return figland_local 

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

        return figland_global 

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

        return figland_imp

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

    return figsphere_local

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

    return figsphere_global

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

    return figsphere_imp

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
                    style = {'display':'inline-block', 'width':'100%'}), #inline-flex,
                    #]),

                ######################################
                #
                #           GRAPH FIGURE 
                #
                ######################################
                html.Div(className = 'eight columns',  #'nine columns', 
                    children = [

                        dcc.Loading(
                            id="loading-2",
                            type="circle",
                            style={'display':'center'},
                            children=
                            html.Div(id="layout-graph",
                                style = {'display':'inline-block', 'width':'100%','height':'80vh'}),
                            ),
                    ]),

                ######################################
                #
                # USER INTERFACE / INTERACTIVE PART
                #
                ######################################
                html.Div(className = 'three columns', #style={'margin-right': '5px'},
                    children = [
                        #----------------------------------------
                        # UPLOAD SECTION
                        #----------------------------------------
                        html.H6('UPLOADS'),
                        #html.P('Upload an edgelist here.'),
                        dcc.Upload(
                                id='upload-data',
                                children=html.Div([
                                    html.A('Upload edgelist here.')
                                ]),
                                style={
                                    'width': '100%',
                                    'height': '32px',
                                    'lineHeight': '32px',
                                    'borderWidth': '1.2px',
                                    'borderStyle': 'dashed',
                                    'borderRadius': '5px',
                                    'textAlign': 'center',
                                    'margin': '0px', 
                                    'font-size':'12px',
                                    'borderColor':'white'
                                },
                                multiple=False# Allow multiple files to be uploaded
                            ),
                        #html.Div(id='output-data-upload'),
                        html.Br(),

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
                    

                        #----------------------------------------
                        # LAYOUTS (local, global, imp, func)
                        #----------------------------------------
                        html.H6('NETWORK LAYOUT'),
                        html.P('Choose one of the listed layouts.'),
                        html.Div(children=[
                            dcc.Dropdown(className='app_dropdown',
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
                            dcc.Dropdown(className='app_dropdown',
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

                        #----------------------------------------
                        # DOWNLOAD SECTION
                        #----------------------------------------
                        html.H6('DOWNLOADS'),
                        html.P('Download Layouts here.'),
                        
                        #html.A(
                        #        id="download-vis", 
                        #        href="", 
                        #        children=[html.Button("Download Image", id="button-vis", n_clicks=0)], 
                        #        target="_blank",
                        #        download="my-figure.pdf"
                        #    ),
                        
                        html.Button('DRAWING', id='button-vis', n_clicks=0,
                                   style={'text-align': 'center', 'width': '100%', 'margin-top': '5px'}),
                                   dcc.Download(id='download-vis'),

                        html.Button('VRNetzer | CSV', id='button-table', n_clicks=0 ,
                                    style={'text-align': 'center', 'width': '100%', 'margin-top': '5px'}),
                        html.Button('3Dprint | OBJ', id='button-obj', n_clicks=0 ,
                                    style={'text-align': 'center', 'width': '100%', 'margin-top': '5px'}),

                    ]),
                        
                #html.Br(),
                #html.Br(),

                html.Div(className = 'two columns', style={'padding-left':'10px'},
                    children = [ 
                        #----------------------------------------
                        # Paper Figures SECTION
                        #----------------------------------------
                        html.Img(src='assets/netimage4.png',
                                style={'height':'220px','width':'100%'
                                }),
                        html.H6('EXPLORE THE HUMAN INTERACTOME'),
                        html.P('View precalculated Layouts of the Human Protein-Protein Interaction Network.'),
                       
                    
                        html.Button('2D PORTRAIT', id='button-ppi-2dport', n_clicks=0 ,   
                                    style={'text-align': 'center', 
                                    'width': '100%', 'margin-top': '5px', #'margin-right':'2px',#'display':'inline-block',
                                   }),
                        dcc.Download(id='download-2dport'),
                        
                        
                        html.Button('3D PORTRAIT', id='button-ppi-3dport', n_clicks=0 ,   
                                    style={'text-align': 'center', 
                                    'width': '100%', 'margin-top': '5px', #'margin-right':'2px',#'display':'inline-block',
                                    }),
                        dcc.Download(id='download-3dport'),
                    
                        
                        html.Button('TOPOGRAPHIC MAP', id='button-ppi-topo', n_clicks=0 ,
                                    style={'text-align': 'center', 
                                    'width': '100%', 'margin-top': '5px', #'display':'inline-block',
                                    }),
                        dcc.Download(id='download-topo'),

                        
                        html.Button('GEODESIC MAP', id='button-ppi-geo', n_clicks=0 ,
                                    style={'text-align': 'center', 
                                    'width': '100%', 'margin-top': '5px', #'margin-left':'2px', #'display':'inline-block',
                                    }),
                        dcc.Download(id='download-geo'),

                    ]), 

                html.Div(className = 'footer',
                    children=[
                        html.P('This visualization app is currently under construction. Feel free to get in touch for bug reports, comments, suggestions via Github/menchelab/cartoGRAPHs'),
                    ]),
            ])
        #])





####################################################################################################################################################################
# TO FIX
####################################################################################################################################################################

def parse_Graph(filename):
    if 'csv' in filename:
            with open(filename, 'r') as edgecsv: # Open the file
                edgereader = csv.reader(edgecsv) # Read the csv
                edges = [tuple(e) for e in edgereader][1:] # Retrieve the data    
                G=nx.Graph() 
                G.add_edges_from(edges)
            return G
    else:
        print('Graph invalid. ')

####################################################################################################################################################################








#########################################
#
#              CALL BACKS               
#
#########################################

#----------------------------------------
# Network Layout + Map
#----------------------------------------
@app.callback(Output('layout-graph', 'children'),

            # button for starting graph
              Input('button-graph-update','n_clicks'),

            # button for upload 
              #[
              Input('upload-data','filename'),
              #Input('upload-data', 'contents')],

            # button for download 
              #Input('button-vis','children'),

            # states of layout and map 
              [State('dropdown-layout-type','value')],
              [State('dropdown-map-type', 'value')],
              )

def update_graph(buttonclicks, inputfile, #inputcontent, download_vis, 
                    #download-csv, download-obj, 
                    layoutvalue, mapvalue):
            
            if buttonclicks == 0:
                G = nx.read_edgelist('input/GPPI_sub_1000.txt')
                fig3D_start = portrait3D_local(G)
                return html.Div(id='layout-graph',children= [
                                            dcc.Graph(
                                                    config={'displayModeBar':False},
                                                    style={'position':'relative','height': '80vh', 'width':'100%'},
                                                    figure=fig3D_start)
                                            ])
                                       
            elif inputfile is None:
                #---------------------------------------
                # Start with this Graph / if no upload
                #---------------------------------------
                G = nx.read_edgelist('input/GPPI_sub_1000.txt')


            elif inputfile:
                    #---------------------------------------
                    # Upload / Input Graph
                    #---------------------------------------
                G = parse_Graph(inputfile)
            

            if buttonclicks:
                ##################
                #
                #  2 d PORTRAIT
                #
                ##################
                if mapvalue == 'fig2D':
                    if layoutvalue == 'local':
                        fig2D_local = portrait2D_local(G)
                        return html.Div(id='layout-graph',children= [
                                                dcc.Graph(
                                                        config={'displayModeBar':False},
                                                        style={'position':'relative','height': '80vh', 'width':'100%'},
                                                        figure=fig2D_local
                                                        ),
                                                    ])

                    elif layoutvalue == 'global':
                        fig2D_global = portrait2D_global(G)
                        return html.Div(id='layout-graph',children= [
                                                dcc.Graph(
                                                        config={'displayModeBar':False},
                                                        style={'position':'relative','height': '80vh', 'width':'100%'},
                                                        figure=fig2D_global
                                                        ),
                                                    ])

                    elif layoutvalue == 'importance':  
                        fig2D_imp = portrait2D_importance(G)
                        return html.Div(id='layout-graph',children= [
                                                dcc.Graph(
                                                        config={'displayModeBar':False},
                                                        style={'position':'relative','height': '80vh', 'width':'100%'},
                                                        figure=fig2D_imp
                                                        ),
                                                    ])

                    # if layoutvalue == 'func':



                ##################
                #
                #  3 D PORTRAIT
                #
                ##################
                elif mapvalue == 'fig3D':
                    if layoutvalue == 'local':
                        fig3D_local = portrait3D_local(G)
                        return html.Div(id='layout-graph', children= [
                                                        dcc.Graph(
                                                                config={'displayModeBar':False},
                                                                style={'position':'relative','height': '80vh', 'width':'100%'},
                                                                figure=fig3D_local
                                                                ),
                                                            ])

                    elif layoutvalue == 'global':
                        fig3D_global = portrait3D_global(G)
                        return html.Div(id='layout-graph',children= [
                                                        dcc.Graph(
                                                                config={'displayModeBar':False},
                                                                style={'position':'relative','height': '80vh', 'width':'100%'},
                                                                figure=fig3D_global
                                                                ),
                                                            ])
                    
                    elif layoutvalue == 'importance':
                        fig3D_imp = portrait3D_importance(G)
                        return html.Div(id='layout-graph',children= [
                                                        dcc.Graph(
                                                                config={'displayModeBar':False},
                                                                style={'position':'relative','height': '80vh', 'width':'100%'},
                                                                figure=fig3D_imp
                                                                ),
                                                            ])

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
                    z_list = list(range(0,len(G.nodes()))) # U P L O A D L I S T  with values if length G.nodes !!! 
                    
                    if layoutvalue == 'local':
                        figland_local = topographic_local(G,z_list)
                        return html.Div(id='layout-graph',children= [
                                                        dcc.Graph(
                                                                config={'displayModeBar':False},
                                                                style={'position':'relative','height': '80vh', 'width':'100%'},
                                                                figure=figland_local
                                                                ),
                                                            ])

                    elif layoutvalue == 'global':
                        figland_global = topographic_global(G,z_list) 
                        return html.Div(id='layout-graph',children= [
                                                dcc.Graph(
                                                        config={'displayModeBar':False},
                                                        style={'position':'relative','height': '80vh', 'width':'100%'},
                                                        figure=figland_global
                                                        ),
                                                    ])
                    
                    elif layoutvalue == 'importance':
                        
                        closeness = nx.closeness_centrality(G)
                        d_clos_unsort  = {}
                        for node, cl in sorted(closeness.items(), key = lambda x: x[1], reverse = 0):
                            d_clos_unsort [node] = round(cl,4)  
                        d_clos = {key:d_clos_unsort[key] for key in G.nodes()}
                        z_list = list(d_clos.values())
                        
                        figland_imp = topographic_importance(G, z_list)

                        return html.Div(id='layout-graph',children= [
                                                dcc.Graph(
                                                        config={'displayModeBar':False},
                                                        style={'position':'relative','height': '80vh', 'width':'100%'},
                                                        figure=figland_imp
                                                        )
                                                    ])

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
                if mapvalue == 'figsphere':
                    radius = dict(zip(G.nodes(),list(range(0,len(G.nodes()))))) # U P L O A D L I S T  with values if length G.nodes !!! 

                    if layoutvalue == 'local':
                        figsphere_local = geodesic_local(G,radius)
                        return html.Div(id='layout-graph',children= [
                                                        dcc.Graph(
                                                                config={'displayModeBar':False},
                                                                style={'position':'relative','height': '80vh', 'width':'100%'},
                                                                figure=figsphere_local
                                                                ),
                                                            ])

                    elif layoutvalue == 'global':  
                        figsphere_global = geodesic_global(G,radius) 
                        return html.Div(id='layout-graph',children= [
                                                dcc.Graph(
                                                        config={'displayModeBar':False},
                                                        style={'position':'relative','height': '80vh', 'width':'100%'},
                                                        figure=figsphere_global
                                                        ),
                                                    ])
                    
                    elif layoutvalue == 'importance':
                        figsphere_imp = geodesic_importance(G,radius)
                        return html.Div(id='layout-graph',children= [
                                                dcc.Graph(
                                                        config={'displayModeBar':False},
                                                        style={'position':'relative','height': '80vh', 'width':'100%'},
                                                        figure=figsphere_imp
                                                        ),
                                                    ])

                    #elif layoutvalue == 'functional':
                    #    figsphere_func = geodesic_func(G,radius)
                    #    return html.Div(id='layout-graph',children= [
                    #                                    dcc.Graph(
                    #                                            config={'displayModeBar':False},
                    #                                            style={'position':'relative','height': '80vh', 'width':'100%'},
                    #                                            figure=figsphere_func
                    #                                            ),
                    #                                        ])



#----------------------------------------
# DOWNLOADS 
#----------------------------------------
'''@app.callback(Output('download-vis', 'data'),
              [
              Input('button-vis','n_clicks'),
              Input('layout-graph', 'children')
              ],
              prevent_initial_call=True,
            )
def get_vis(n_clicks, figure):
    return dcc.send_file(
        figure)
'''
'''
# downloading figure
@app.callback(Output('download-link', 'href'),
              [Input('button-vis', 'n_clicks')])
def update_link(nclicks):
    if nclicks > 0:
        print('DEBUG: link updated, nclicks=' + str(nclicks))
        return '/dash/urlToDownload'

@app.server.route('/dash/urlToDownload')
def download_file():
    
    pathToWrite = "figure.html"
    print('DEBUG(inroute): looking for file at ' + pathToWrite)
    return send_from_directory(
                               'figure.html',
                               as_attachment=True,
                               cache_timeout=0)
'''



server = app.server
if __name__ == '__main__':
    app.run_server(debug=True)







#---------------------------------------
# PPI precalculated 
#---------------------------------------
            
'''
G_PPI = nx.read_edgelist('input/ppi_elist.txt')  
                edge_width = 0.2
                edge_opac = 0.4
                edge_colordark = '#666666'
                node_size = 1.0
                d_gene_sym = load_genesymbols(G_PPI, 'human')
                l_features_PPI = list(d_gene_sym.values()) 
                
                df3D_PPI = pd.read_csv('input/3D_global_layout_human.csv', header=None)
                df3D_PPI.columns = ['id','x','y','z','r','g','b','a','namespace']

                ids3D_PPI = [str(i) for i in list(df3D_PPI['id'])]
                x_3D_PPI = list(df3D_PPI['x'])
                y_3D_PPI = list(df3D_PPI['y'])
                z_3D_PPI = list(df3D_PPI['z'])
                posG_3D_PPI = dict(zip(ids3D_PPI,zip(x_3D_PPI,y_3D_PPI,z_3D_PPI)))

                r3D_PPI = list(df3D_PPI['r'])
                g3D_PPI = list(df3D_PPI['g'])
                b3D_PPI = list(df3D_PPI['b'])
                a3D_PPI = list(df3D_PPI['a'])
                colours3D_PPI = list(zip(r3D_PPI,g3D_PPI,b3D_PPI,a3D_PPI))

                umap3D_nodes_PPI = get_trace_nodes_3D(posG_3D_PPI, l_features_PPI, colours3D_PPI, node_size) #size3D)
                umap3D_edges_PPI = get_trace_edges_3D(G_PPI, posG_3D_PPI, edge_colordark, edge_opac, edge_width) 
                umap3D_data_PPI = [umap3D_edges_PPI, umap3D_nodes_PPI]
                fig3D_PPI = plot3D_app(umap3D_data_PPI) 

                return html.Div(id='layout-graph',children= [
                                                                dcc.Graph(
                                                                        config={'displayModeBar':False},
                                                                        style={'position':'relative','height': '80vh', 'width':'100%'},
                                                                        figure=fig3D_PPI
                                                                        ),
                                                                    ])

                                                                    
#-------------------------------
# Topographic Map PPI 
#-------------------------------
dfland_PPI = pd.read_csv('input/topographic_global_layout_human.csv', header=None)
dfland_PPI.columns = ['id','x','y','z','r','g','b','a','namespace']

idsland_PPI = [str(i) for i in list(dfland_PPI['id'])]
x_land_PPI = list(dfland_PPI['x'])
y_land_PPI = list(dfland_PPI['y'])
z_land_PPI = list(dfland_PPI['z'])
posG_land_PPI = dict(zip(idsland_PPI,zip(x_land_PPI,y_land_PPI,z_land_PPI)))

rland_PPI = list(dfland_PPI['r'])
gland_PPI = list(dfland_PPI['g'])
bland_PPI = list(dfland_PPI['b'])
aland_PPI = list(dfland_PPI['a'])

coloursland_PPI = list(zip(rland_PPI,gland_PPI,bland_PPI,aland_PPI))

umapland_nodes_PPI = get_trace_nodes_3D(posG_land_PPI, l_features_PPI, coloursland_PPI, node_size)# size3d)
umapland_edges_PPI = get_trace_edges_3D(G_PPI, posG_land_PPI, edge_colordark, opac = 0.4)
umapland_data_PPI= [umapland_edges_PPI, umapland_nodes_PPI]

figland_PPI = pgo.Figure()
for i in umapland_data_PPI:
    figland_PPI.add_trace(i)

figland_PPI.update_layout(
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

                
#-------------------------------
# Geodesic Map PPI 
#-------------------------------
dfsphere_PPI = pd.read_csv('input/geodesic_global_layout_human.csv', header=None)
dfsphere_PPI.columns = ['id','x','y','z','r','g','b','a','namespace']

idssphere_PPI = [str(i) for i in list(dfsphere_PPI['id'])]
x_sphere_PPI = list(dfsphere_PPI['x'])
y_sphere_PPI = list(dfsphere_PPI['y'])
z_sphere_PPI = list(dfsphere_PPI['z'])
posG_sphere_PPI = dict(zip(idssphere_PPI,zip(x_sphere_PPI,y_sphere_PPI,z_sphere_PPI)))

rsphere_PPI = list(dfsphere_PPI['r'])
gsphere_PPI = list(dfsphere_PPI['g'])
bsphere_PPI = list(dfsphere_PPI['b'])
asphere_PPI = list(dfsphere_PPI['a'])

colourssphere_PPI = list(zip(rsphere_PPI,gsphere_PPI,bsphere_PPI,asphere_PPI))

umapsphere_nodes_PPI = get_trace_nodes_3D(posG_sphere_PPI, l_features_PPI, colours, node_size) # size3d)
umapsphere_edges_PPI = get_trace_edges_3D(G, posG_sphere_PPI, edge_colordark, opac=0.4)
umapsphere_data_PPI = [umapsphere_edges_PPI, umapsphere_nodes_PPI]

figsphere_PPI = pgo.Figure()
for i in umapsphere_data_PPI:
    figsphere_PPI.add_trace(i)

figsphere_PPI.update_layout(
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

'''