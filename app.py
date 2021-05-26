from networkx.generators import line
from app_main import *


# Initialise the app
app = dash.Dash(__name__)




##################################################################################
##################################################################################
#
#                          P R E P   F O R   A P P                   
#
##################################################################################
##################################################################################


'''#------------------------------
# PLACEHOLDER GRAPH   
#------------------------------ 
G=nx.read_edgelist('input/GPPI_sub_1000.txt')

closeness = nx.closeness_centrality(G)
d_clos = {}
for node, cl in sorted(closeness.items(), key = lambda x: x[1], reverse = 1):
    d_clos[node] = round(cl,4)
d_nodecol = d_clos 
col_pal = 'RdYlBu'
d_colours = color_nodes_from_dict(G, d_nodecol, palette = col_pal)
colours = list(d_colours.values())
l_feat = list(G.nodes())

scale_factor3D = 0.025
node_size_deg = list(draw_node_degree_3D(G, scale_factor3D).values())'''


#########################################
#         
#          GENERAL PARAMETERS           
#
#########################################

#------------------------------
# EDGES
#------------------------------
edge_width = 0.9
edge_opac = 0.9
edge_colorlight = '#666666' 
edge_colordark = '#555555' 

#------------------------------
# NODES
#------------------------------ 
node_size = 1.0
opacity_nodes = 0.9
node_edge_col = '#696969'

nodesglow_diameter = 20.0
nodesglow_transparency = 0.05 # 0.01

#------------------------------
# UMAP SETTINGS
#------------------------------ 
n_neighbors = 20 
spread = 0.9
min_dist = 0
metric='cosine'


#------------------------------
# GRAPH
#------------------------------ 
G = nx.Graph()
if len(G.nodes()) == 0: 
    ##########################################
    #         
    #         IF No Button clicks FIGURE          
    #
    ##########################################

    G_start=nx.read_edgelist('input/GPPI_sub_50.txt')
    G = G_start

    closeness = nx.closeness_centrality(G)
    d_clos_unsort  = {}
    for node, cl in sorted(closeness.items(), key = lambda x: x[1], reverse = 0):
        d_clos_unsort [node] = round(cl,4)  
    col_pal = 'RdYlBu'
    d_clos = {key:d_clos_unsort[key] for key in G.nodes()}
    d_nodecol = d_clos
    d_colours = color_nodes_from_dict(G, d_nodecol, palette = col_pal)
    colours = list(d_colours.values())
    node_size = 3.0
    l_feat = list(G_start.nodes())
    A_start = nx.adjacency_matrix(G_start, nodelist=list(G.nodes()))
    DM_m_start = pd.DataFrame(rnd_walk_matrix2(A_start,0.9,1,len(G_start))).T
    DM_m_start.index=list(G_start.nodes())
    DM_m_start.columns=list(G_start.nodes())

    embed3D_start = embed_umap_3D(DM_m_start,n_neighbors,spread,min_dist,metric)
    posG_3D_start = get_posG_3D(list(G_start.nodes()),embed3D_start) 
    umap3D_nodes_start = get_trace_nodes_3D(posG_3D_start, l_feat, colours, node_size)
    umap3D_edges_start = get_trace_edges_3D(G_start, posG_3D_start, edge_colordark, opac=edge_opac, linewidth=edge_width) 
    umap3D_data_start = [umap3D_edges_start, umap3D_nodes_start]

    fig3D_start = pgo.Figure()
    for i in umap3D_data_start:
        fig3D_start.add_trace(i)

    fig3D_start.update_layout(
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


#########################################
#         
#          NETWORK LAYOUTS         
#
#########################################

##### local #####
A = nx.adjacency_matrix(G, nodelist=list(G.nodes()))
M_adj = A.toarray()
DM_adj = pd.DataFrame(M_adj)
DM_adj.index=list(G.nodes())
DM_adj.columns=list(G.nodes())

##### global #####
DM_m = pd.DataFrame(rnd_walk_matrix2(A,0.9,1,len(G))).T
DM_m.index=list(G.nodes())
DM_m.columns=list(G.nodes())

#### importance ####
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
feature_df = pd.DataFrame.from_dict(feature_dict_sorted, orient = 'index', columns = ['degs', 'clos', 'betw'])
DM_imp = feature_df

#### functional #### 





##########################################
#         
#          3D PORTRAITS            
#
#########################################

#------------------------------
# LOCAL 
#------------------------------  
embed3D_local = embed_umap_3D(DM_adj,n_neighbors,spread,min_dist,metric)
posG_3D_local = get_posG_3D(list(G.nodes()),embed3D_local) 
umap3D_nodes_local = get_trace_nodes_3D(posG_3D_local, l_feat, colours, node_size)
umap3D_edges_local = get_trace_edges_3D(G, posG_3D_local, edge_colordark, opac=edge_opac, linewidth=edge_width) 
umap3D_data_local = [umap3D_edges_local, umap3D_nodes_local]

fig3D_local = pgo.Figure()
for i in umap3D_data_local:
    fig3D_local.add_trace(i)

fig3D_local.update_layout(
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

#------------------------------
# GLOBAL
#------------------------------  
embed3D_global = embed_umap_3D(DM_m,n_neighbors,spread,min_dist,metric)
posG_3D_global = get_posG_3D(list(G.nodes()),embed3D_global) 
umap3D_nodes_global = get_trace_nodes_3D(posG_3D_global, l_feat, colours, node_size)
umap3D_edges_global = get_trace_edges_3D(G, posG_3D_global, edge_colordark, opac=edge_opac, linewidth=edge_width) 
umap3D_data_global = [umap3D_edges_global, umap3D_nodes_global]

fig3D_global = pgo.Figure()
for i in umap3D_data_global:
    fig3D_global.add_trace(i)

fig3D_global.update_layout(
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

#------------------------------
# IMPORTANCE 
#------------------------------  
embed3D_imp = embed_umap_3D(DM_imp,n_neighbors,spread,min_dist,metric)
posG_3D_imp = get_posG_3D(list(G.nodes()),embed3D_imp) 
umap3D_nodes_imp = get_trace_nodes_3D(posG_3D_imp, l_feat, colours, node_size)
umap3D_edges_imp = get_trace_edges_3D(G, posG_3D_imp, edge_colordark, opac=edge_opac, linewidth=edge_width) 
umap3D_data_imp = [umap3D_edges_imp, umap3D_nodes_imp]

fig3D_imp = pgo.Figure()
for i in umap3D_data_imp:
    fig3D_imp.add_trace(i)

fig3D_imp.update_layout(
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
#------------------------------
# FUNCTIONAL 
#------------------------------  





##########################################
#         
#          TOPOGRAPHIC MAP             
#
#########################################

#------------------------------
# LOCAL 
#------------------------------ 
z_list = list(d_clos_sorted.values()) # U P L O A D L I S T  with values !!! 
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

figland_local = pgo.Figure()
for i in umapland_data_local:
    figland_local.add_trace(i)

figland_local.update_layout(
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

#------------------------------
# GLOBAL
#------------------------------  
z_list = list(d_clos_sorted.values()) # U P L O A D L I S T  with values !!! 
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

figland_global = pgo.Figure()
for i in umapland_data_global:
    figland_global.add_trace(i)

figland_global.update_layout(
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

#------------------------------
# IMPORTANCE
#------------------------------  
z_list = list(d_clos_sorted.values()) # U P L O A D L I S T  with values !!! 
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

figland_imp = pgo.Figure()
for i in umapland_data_imp:
    figland_imp.add_trace(i)

figland_imp.update_layout(
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

#------------------------------
# FUNCTIONAL 
#------------------------------  




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
                    html.Img(src='assets/cartoGraphs_logo_long.png',style={'height':'80px'}),
                    ],
                    style = {'display':'inline-block', 'width':'100%'}, #inline-flex
                ),
                ######################################
                #
                #           GRAPH FIGURE 
                #
                ######################################
                html.Div(className = 'nine columns', 
                children = [
                    html.Div(
                        id='layout-graph'
                        ),
                    html.P('This visualization app is currently under construction. Feel free to get in touch for bug reports, comments, suggestions via Github/menchelab/cartoGRAPHs',
                    style = {'display':'bottom'}),
                ]),

                ######################################
                #
                # USER INTERFACE / INTERACTIVE PART
                #
                ######################################
                html.Div(className = 'three columns',
                children = [
                    html.H6('UPLOADS'),
                    html.P('Upload an edgelist here.'),
                    dcc.Upload(
                            id='upload-edgelist',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select Files')
                            ]),
                            style={
                                'width': '100%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '0px', 
                            },
                            # Allow multiple files to be uploaded
                            multiple=True
                        ),
                        html.Div(id='output-data-upload'),

                    html.Br(),

                    #----------------------------------------
                    # LAYOUTS (local, global, imp, func)
                    #----------------------------------------
                    html.H6('NETWORK LAYOUT'),
                    html.P('Choose of the following layouts.'),
                    html.Div(children=[
                        dcc.Dropdown(
                            id='dropdown-layout-type',
                            options=[
                                {'label': 'local', 'value': 'local'},
                                {'label': 'global', 'value': 'global'},
                                {'label': 'importance', 'value': 'importance'},
                                {'label': 'functional', 'value': 'functional'},
                            ],
                            placeholder="Select a Network Layout.", 
                            style={'color':'#000000'} #font color for dropdown menu
                            )
                    ]),
                
                    #----------------------------------------
                    # MAP CATEGORY (2D,3D,topo,geo)
                    #----------------------------------------
                    html.H6('NETWORK MAP CATEGORY'),
                    html.P('Choose of the following map categories.'),
                    html.Div(children=[
                        dcc.Dropdown(
                            id='dropdown-map-type',
                            options=[
                                {'label': '2D Portrait', 'value': 'fig2D'},
                                {'label': '3D Portrait', 'value': 'fig3D'},
                                {'label': 'Topographic Map', 'value': 'figland'},
                                {'label': 'Geodesic Map', 'value': 'figsphere'},
                            ],
                            placeholder="Select a Layout Map.", 
                            style={'color':'#000000'} #font color for dropdown menu
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
                    html.Button('2D | PNG', id='button-png', n_clicks=0 ,
                                style={'text-align': 'center','width': '100%', 'margin-top': '5px'}),
                    html.Button('3D | HTML', id='button-html', n_clicks=0 ,   
                                style={'text-align': 'center', 'width': '100%', 'margin-top': '5px'}),
                    html.Button('VRNetzer | TABLE', id='button-table', n_clicks=0 ,
                                style={'text-align': 'center', 'width': '100%', 'margin-top': '5px'}),
                    html.Button('3Dprint | OBJ', id='button-obj', n_clicks=0 ,
                                style={'text-align': 'center', 'width': '100%', 'margin-top': '5px'}),
                    
                    html.Br(),
                    html.Br(),

                    #----------------------------------------
                    # Paper Figures SECTION
                    #----------------------------------------
                    
                    html.H6('DRAW HUMAN INTERACTOME'),
                    html.Button('3D PORTRAIT', id='button-ppi-portrait', n_clicks=0 ,   
                                style={#'text-align': 'center', 
                                'width': '33%', 'margin-top': '2px', 'margin-right':'2px','display':'inline-block',
                                'font-size':'9px'}),
                    html.Button('TOPOGRAPHIC MAP', id='button-ppi-topo', n_clicks=0 ,
                                style={#'text-align': 'center', 
                                'width': '33%', 'margin-top': '2px', 'display':'inline-block',
                                'font-size':'9px'}),
                    html.Button('GEODESIC MAP', id='button-ppi-geo', n_clicks=0 ,
                                style={#'text-align': 'center', 
                                'width': '33%', 'margin-top': '2px', 'margin-left':'2px', 'display':'inline-block'
                                'font-size':'9px', 'font-color':'white'})
                    ]),       

            ])


#########################################
#
#              CALL BACKS               
#
#########################################

#----------------------------------------
#         Graph input Callback           
#----------------------------------------

@app.callback(Output('output-data-upload', 'children'),
              Input('upload-edgelist', 'contents'))
def parse_edgelist(filename):
    try:
        if '.txt' in filename:
            with open('r') as filname:
                textfile = filename.read()
                return textfile

    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])


#----------------------------------------
# Network Layout + Map
#----------------------------------------
@app.callback(Output('layout-graph', 'children'),
              Input('button-graph-update','n_clicks'),
              State('dropdown-map-type', 'value'),
              State('dropdown-layout-type','value'))

def update_layout(buttonclicks, mapvalue, layoutvalue):
            if buttonclicks == 0:
                return html.Div(id='layout-graph',children= [
                                        dcc.Graph(
                                                config={'displayModeBar':False},
                                                style={'position':'relative','height': '80vh', 'width':'100%'},
                                                figure=fig3D_start
                                                ),
                                            ])
            else:
                # 3D PORTRAIT OPTIONS 
                #----------------------------------------
                if mapvalue == 'fig3D' and layoutvalue == 'local':
                        return html.Div(id='layout-graph',children= [
                                        dcc.Graph(
                                                config={'displayModeBar':False},
                                                style={'position':'relative','height': '80vh', 'width':'100%'},
                                                figure=fig3D_local
                                                ),
                                            ])
                elif mapvalue == 'fig3D' and layoutvalue == 'global':
                        return html.Div(id='layout-graph',children= [
                                        dcc.Graph(
                                                config={'displayModeBar':False},
                                                style={'position':'relative','height': '80vh', 'width':'100%'},
                                                figure=fig3D_global
                                                ),
                                            ])
                elif mapvalue == 'fig3D' and layoutvalue == 'importance':
                        return html.Div(id='layout-graph',children= [
                                        dcc.Graph(
                                                config={'displayModeBar':False},
                                                style={'position':'relative','height': '80vh', 'width':'100%'},
                                                figure=fig3D_imp
                                                ),
                                            ])
                    
                # TOPOGRAPHIC OPTIONS 
                #----------------------------------------
                elif mapvalue == 'figland' and layoutvalue == 'local':
                        return html.Div(id='layout-graph',children= [
                                        dcc.Graph(
                                                config={'displayModeBar':False},
                                                style={'position':'relative','height': '80vh', 'width':'100%'},
                                                figure=figland_local
                                                ),
                                            ])
                elif mapvalue == 'figland' and layoutvalue == 'global':
                        return html.Div(id='layout-graph',children= [
                                        dcc.Graph(
                                                config={'displayModeBar':False},
                                                style={'position':'relative','height': '80vh', 'width':'100%'},
                                                figure=figland_global
                                                ),
                                            ])
                elif mapvalue == 'figland' and layoutvalue == 'importance':
                        return html.Div(id='layout-graph',children= [
                                        dcc.Graph(
                                                config={'displayModeBar':False},
                                                style={'position':'relative','height': '80vh', 'width':'100%'},
                                                figure=figland_imp
                                                ),
                                            ])

                '''# GEODESIC OPTIONS 
                #----------------------------------------
                elif mapvalue == 'figsphere' and layoutvalue == 'local':
                        return html.Div(id='layout-graph',children= [
                                        dcc.Graph(
                                                config={'displayModeBar':False},
                                                style={'position':'relative','height': '80vh', 'width':'100%'},
                                                figure=figsphere_local
                                                ),
                                            ])
                elif mapvalue == 'figsphere' and layoutvalue == 'global':
                        return html.Div(id='layout-graph',children= [
                                        dcc.Graph(
                                                config={'displayModeBar':False},
                                                style={'position':'relative','height': '80vh', 'width':'100%'},
                                                figure=figsphere_global
                                                ),
                                            ])
                elif mapvalue == 'figsphere' and layoutvalue == 'importance':
                        return html.Div(id='layout-graph',children= [
                                        dcc.Graph(
                                                config={'displayModeBar':False},
                                                style={'position':'relative','height': '80vh', 'width':'100%'},
                                                figure=figsphere_imp
                                                ),
                                            ])'''
           

server = app.server
if __name__ == '__main__':
    app.run_server(debug=False)