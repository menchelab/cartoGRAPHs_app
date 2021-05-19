from app_functions import *


# Initialise the app
app = dash.Dash(__name__)


#########################################
#              FILE INPUT               #
#########################################

# GRAPH input ( example for now )
organism = 'Yeast'
data = pickle.load( open('data/BIOGRID-ORGANISM-Saccharomyces_cerevisiae_S288c-3.5.185.mitab.pickle', "rb" ) )
filter_score = data[(data['Interaction Types'] == 'psi-mi:"MI:0407"(direct interaction)')]
g = nx.from_pandas_edgelist(filter_score, '#ID Interactor A', 'ID Interactor B')
g.remove_edges_from(nx.selfloop_edges(g)) #remove self loop
G = g.subgraph(max(nx.connected_components(g), key=len)) # largest connected component (lcc)
posG_entrez = []
for k in G.nodes():
    posG_entrez.append(k[22:])

# List of Features for hover info 
df_centralities = pd.read_csv('data/Features_centralities_Dataframe_'+organism+'.csv', index_col=0)
d_deghubs = dict(zip(G.nodes(),df_centralities['degs']))
d_clos = dict(zip(G.nodes(), df_centralities['clos']))
d_betw = dict(zip(G.nodes(), df_centralities['betw']))
d_eigen = dict(zip(G.nodes(), df_centralities['eigen']))
d_centralities = dict(zip(list(G.nodes),zip(d_deghubs.values(),d_clos.values(),d_betw.values(),d_eigen.values())))

l_features = []
for i in d_centralities.items():
    k=list(i)
    l_features.append(k)



#########################################
#        SET VISUAL PROPERTIES          #
#########################################

# Node, Edge colors
edge_width = 0.2
edge_colorlight = 'lightgrey'
edge_colordark = 'dimgrey'
opacity_nodes = 1.0

# Node sizes 
scalef= 0.05
size = list(draw_node_degree(G, scalef).values())

scalef= 0.05
size3D = list(draw_node_degree_3D(G, scalef).values())

# ----------------------------------------------------        
# COLOUR PARAMETER
# ----------------------------------------------------        
col_method = 'clos'
d_to_be_col = d_clos # dict sorted by dict.values (that way the biggest value matches darkest colour of palette)
colours = color_nodes_from_dict(G, d_to_be_col, col_method)


#########################################
#          MODIFY DATA FOR PLOT         #
#########################################
# for now: read csv from VR compatible layout 

# -----------------
# 2D PORTRAIT 
# -----------------
df2D = pd.read_csv('data/NEON_portrait2Dumap_Yeast.csv', header=None)
df2D.columns = ['id','x','y','z','r','g','b','a','namespace']

ids2D = list(G.nodes())
x_2D = list(df2D['x'])
y_2D = list(df2D['y'])
posG_2D = dict(zip(ids2D,zip(x_2D,y_2D)))

umap2D_nodes = get_trace_nodes_2D(posG_2D, l_features, colours, size) 
umap2D_edges = get_trace_edges_2D(G, posG_2D, edge_colordark, opac=0.4) 
umap2D_data = [umap2D_edges, umap2D_nodes]

fig2D = pgo.Figure()
for i in umap2D_data:
    fig2D.add_trace(i)

fig2D.update_layout(
                    margin=dict(l=0, r=0, t=0, b=0),
                    template='plotly_white', 
                    plot_bgcolor='white',
                    font_color="white",
                    showlegend=False, autosize = True,#width=1600, height=800,
                    xaxis = {'showgrid':False,'zeroline':False,},
                    yaxis = {'showgrid':False,'zeroline':False}
                )   

# -----------------
# 3D PORTRAIT 
# -----------------
df3D = pd.read_csv('data/NEON_portrait3Dumap_Yeast.csv', header=None)
df3D.columns = ['id','x','y','z','r','g','b','a','namespace']

ids3D = list(G.nodes())
x_3D = list(df3D['x'])
y_3D = list(df3D['y'])
z_3D = list(df3D['z'])
posG_3D = dict(zip(ids3D,zip(x_3D,y_3D,z_3D)))

umap3D_nodes = get_trace_nodes_3D(posG_3D, l_features, colours, 1.2) #size3D)
umap3D_edges = get_trace_edges_3D(G, posG_3D, edge_colordark, opac=0.4) 
umap3D_data = [umap3D_edges, umap3D_nodes]

fig3D = pgo.Figure()
for i in umap3D_data:
    fig3D.add_trace(i)

fig3D.update_layout(
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
                    #annotations=annotations,
                ))   


# -----------------
# LANDSCAPE 
# -----------------
dfland = pd.read_csv('data/NEON_landscapeumap_Yeast.csv', header=None)
dfland.columns = ['id','x','y','z','r','g','b','a','namespace']

idsland = list(G.nodes())
x_land = list(dfland['x'])
y_land = list(dfland['y'])
z_land = list(dfland['z'])
posG_land = dict(zip(idsland,zip(x_land,y_land,z_land)))

umapland_nodes = get_trace_nodes_3D(posG_land, l_features, colours, 1)# size3d)
umapland_edges = get_trace_edges_3D(G, posG_land, edge_colordark, opac=0.4)
umapland_data = [umapland_edges, umapland_nodes]

figland = pgo.Figure()
for i in umapland_data:
    figland.add_trace(i)

figland.update_layout(
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
                    #annotations=annotations,
                ))   


# -----------------
# SPHERE
# -----------------
dfsphere = pd.read_csv('data/NEON_sphereumap_Yeast.csv', header=None)
dfsphere.columns = ['id','x','y','z','r','g','b','a','namespace']

idssphere = list(G.nodes())
x_sphere = list(dfsphere['x'])
y_sphere = list(dfsphere['y'])
z_sphere = list(dfsphere['z'])
posG_sphere = dict(zip(idssphere,zip(x_sphere,y_sphere,z_sphere)))

umapsphere_nodes = get_trace_nodes_3D(posG_sphere, l_features, colours, 1)# size3d)
umapsphere_edges = get_trace_edges_3D(G, posG_sphere, edge_colordark, opac=0.4)
umapsphere_data = [umapsphere_edges, umapsphere_nodes]

figsphere = pgo.Figure()
for i in umapsphere_data:
    figsphere.add_trace(i)

figsphere.update_layout(
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
                    #annotations=annotations,
                ))  



#########################################
#                  APP                  #
#########################################

app.layout = html.Div(
        id='app__banner',
        #style={'display':'inline'},
        children=[
                html.Div(
                className="app__banner",
                children=[ 
                    html.Img(src="assets/neon_logo_nobrackets.png",style = {'display':'inline', 'height':'100px'}),
                    html.H1(" | A Framework for Multidimensional Network Visualization", style={'margin-left':'20px','margin-top':'38px'}),
                    ],
                    style = {'display':'inline-flex'},
                ),
                html.Div(className = 'nine columns', 
                children = [
                    # Network picture
                    html.Div(
                        id='layout-graph'
                        ),
                    html.P('This visualization app is currently under construction. @MencheLab'),
                ]),
                html.Div(className = 'three columns', 
                children = [
                    
                    # INPUT: Graph 
                    # html.H6('Graph'),
                    # html.P('Upload a .txt file i.e. edge list.'),
                    # html.Div(children=[
                    #     dcc.Input(
                    #         id='input-edgelistgraph',
                    #         # type =
                    #         placeholder = 'input type: edgelist',
                    #         )
                    #     ]),
                    # html.Br(),
                    # html.Br(),

                    # INPUT: feature matrix
                    html.H6('Feature Matrix'),
                    html.P('Upload a dataframe, containing network nodes with a selection of features.'),
                    html.Div(children=[
                        dcc.Input(
                            id='input-featurematrix',
                            #type = 
                            placeholder = 'input type: matrix',
                            )
                        ]),
                    html.Br(),
                    html.Br(),
                    
                    # INPUT: Network layout type
                    html.H6('Network Layout Type'),
                    html.P('Select one of four provided layout typologies.'),
                    html.Div(children=[
                        dcc.Dropdown(
                            id='dropdown-layout-type',
                            options=[
                                {'label': '2D Portrait', 'value': 'fig2D'},
                                {'label': '3D Portrait', 'value': 'fig3D'},
                                {'label': 'Landscape', 'value': 'figland'},
                                {'label': 'Spherescape', 'value': 'figsphere'},
                            ],
                            placeholder="Select a Layout Type", 
                            style={'color':'#000000'} #font color for dropdown menu
                            )
                    ]),
                    html.Br(),
                    html.Br(),
                    html.H6('Visual Properties'),
                    html.P('Choose one of the following visualization options, e.g. node colors and size parameters.'),
                    html.Br(),
                    html.P('Node Colour'),
                    html.Div(children=[
                        dcc.Dropdown(
                            options=[
                                {'label': 'Degree Centrality', 'value': 'deg'},
                                {'label': 'Closeness Centrality', 'value': 'clos'},
                                {'label': 'Biological Functions', 'value': 'biofunc'},
                                {'label': 'Specific Gene List', 'value': 'genelist'},
                            ],
                            placeholder="Select a node colouring parameter",
                            style={'color':'#000000'} #font color for dropdown menu
                            )
                    ]),       
                    html.Br(),
                    html.P('Node Size'),
                    html.Div(children=[
                        dcc.Dropdown(
                            options=[
                                {'label': 'Degree Centrality', 'value': 'deg'},
                                {'label': 'All same', 'value': 'same'},
                                {'label': 'Specific Gene List', 'value': 'genelist'},
                            ],
                            placeholder="Select a node size parameter",
                            style={'color':'#000000'} #font color for dropdown menu
                            )
                    ]),         
                ])
            ])



#########################################
#              CALL BACKS               #
#########################################

# Network Layout Typology 
@app.callback(Output('layout-graph', 'children'),
              [Input('dropdown-layout-type', 'value')]
              )
def update_layout(value):
        if value is None:
            return html.Div(id='layout-graph', children = [
                            dcc.Graph(
                                    config={'displayModeBar':False},
                                    style={'position':'relative','height': '80vh', 'width':'100%'},
                                    figure=fig3D
                                    ),
            ])
        elif value == 'fig2D':
            return html.Div(id='layout-graph',children= [
                            dcc.Graph(
                                    config={'displayModeBar':False},
                                    style={'position':'relative','height': '80vh', 'width':'100%'},
                                    figure=fig2D
                                    ),
                                ])

        elif value == 'fig3D':
            return html.Div(id='layout-graph',children= [
                            dcc.Graph(
                                    config={'displayModeBar':False},
                                    style={'position':'relative','height': '80vh', 'width':'100%'},
                                    figure=fig3D
                                    ),
                                ])
        elif value == 'figland':
            return html.Div(id='layout-graph',children= [
                            dcc.Graph(
                                    config={'displayModeBar':False},
                                    style={'position':'relative','height': '80vh', 'width':'100%'},
                                    figure=figland
                                    ),
                                ])
                        
        elif value == 'figsphere':
            return html.Div(id='layout-graph',children= [
                            dcc.Graph(
                                    config={'displayModeBar':False},
                                    style={'position':'relative','height': '80vh', 'width':'100%'},
                                    figure=figsphere
                                    ),
                                ])

# Run the app
server = app.server
if __name__ == '__main__':
    app.run_server(debug=False)