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
scalef= 0.2
size = list(draw_node_degree(G, scalef).values())

scalef= 0.05
size3D = list(draw_node_degree_3D(G, scalef).values())

# ----------------------------------------------------        
# COLOUR PARAMETER
# ----------------------------------------------------        
color_method = 'clos'
d_to_be_coloured = d_clos # dict sorted by dict.values (that way the biggest value matches darkest colour of palette)

# Colouring
colour_groups = set(d_to_be_coloured.values())
colour_count = len(colour_groups)
pal = sns.color_palette('YlOrRd', colour_count)
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
l_col = list(d_node_colour_sorted.values())
colours = l_col




#########################################
#          MODIFY DATA FOR PLOT         #
#########################################

# -----------------
# 3D PORTRAIT 
# -----------------

# for now: read csv from VR compatible layout 
df3D = pd.read_csv('data/NEON_portrait3Dumap_Yeast.csv', header=None)
df3D.columns = ['id','x','y','z','r','g','b','a','namespace']

ids3D = list(G.nodes())
x_3D = list(df3D['x'])
y_3D = list(df3D['y'])
z_3D = list(df3D['z'])
posG_3D = dict(zip(ids3D,zip(x_3D,y_3D,z_3D)))

umap3D_nodes = get_trace_nodes(posG_3D, l_features, colours, 1.2) #size3D)
umap3D_edges = get_trace_edges(G, posG_3D, edge_colordark, opac=0.4) 
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
# for now: read csv from VR compatible layout 
dfland = pd.read_csv('data/NEON_landscapeumap_Yeast.csv', header=None)
dfland.columns = ['id','x','y','z','r','g','b','a','namespace']

idsland = list(G.nodes())
x_land = list(dfland['x'])
y_land = list(dfland['y'])
z_land = list(dfland['z'])
posG_land = dict(zip(idsland,zip(x_land,y_land,z_land)))

umapland_nodes = get_trace_nodes(posG_land, l_features, colours, 1)# size3d)
umapland_edges = get_trace_edges(G, posG_land, edge_colordark)
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
                    #html.Img(src="assets/prelim_logo2.png",style = {'display':'inline', 'width':'80px','height':'80px'}),
                    html.Img(src="assets/neon_logo_nobrackets.png",style = {'display':'inline', 'height':'100px'}),
                    html.H1(" | A Framework for Multidimensional NEtwork VisualizatiON", style={'margin-left':'20px','margin-top':'38px'}),
                    ],
                    style = {'display':'inline-flex'},
                ),
                html.Div(className = 'nine columns', 
                children = [
                    # Network picture
                    html.Div(
                        id='layout-graph',
                        #children = [
                        #    dcc.Graph(
                        #        id='layout-graph',
                        #        config={'displayModeBar':False},
                        #        style={'position':'relative','height': '80vh', 'width':'100%'},
                        #        #figure=fig3D
                        #        ),
                        #]
                        ),
                    html.P('This visualization app is currently under construction. @MencheLab'),
                ]),
                html.Div(className = 'three columns', 
                children = [
                    html.H6('Feature Matrix'),
                    html.P('Upload a dataframe, containing network nodes with a selection of features.'),
                    html.Br(),
                    html.Br(),
                    
                    html.H6('Network Layout Type'),
                    html.P('Select one of four provided layout typologies.'),
                    html.Div(children=[
                        dcc.Dropdown(
                            id='dropdown-layout-type',
                            options=[
                                {'label': '2D Portrait', 'value': 'fig2D'},
                                {'label': '3D Portrait', 'value': 'fig3D'},
                                {'label': 'Landscape', 'value': 'figLand'},
                                {'label': 'Spherescape', 'value': 'figSphere'},
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

@app.callback(Output('layout-graph', 'children'),
             [Input('dropdown-layout-type', 'value')]
             )
def update_layout(value):
    if value is None:
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

    elif value == 'landscape':

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



    #else:
    #    return html.Div(
    #                    children = [
    #                        dcc.Graph(
    #                            id='layout-graph',
    #                            config={'displayModeBar':False},
    #                            style={'position':'relative','height': '80vh', 'width':'100%'},
    #                            figure=value
    #                            ),
    #                    ])


# Run the app
server = app.server
if __name__ == '__main__':
    app.run_server(debug=True)