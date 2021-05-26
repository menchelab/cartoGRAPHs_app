from app_functions import *


# Initialise the app
app = dash.Dash(__name__)


#########################################
#              FILE INPUT               #
#########################################

# define netlayout (use a dropdown)
netlayout = 'global' 





#####################
# SELECT AN ORGANISM 
#####################
organism = 'human' #'yeast'

G = load_graph(organism)
d_genesym = load_genesymbols(G, organism)
l_sym = list(d_genesym.values())

d_centralities = load_centralities(G, organism)
df_centralities = pd.DataFrame(d_centralities).T
essential_genes,non_ess_genes,notdefined_genes = load_essentiality(G, organism)
d_gene_sym = load_genesymbols(G, organism)


# List of Features for hover info 
df_centralities = pd.read_csv('input/Features_centralities_Dataframe_'+organism+'.csv', index_col=0)
d_deghubs = dict(G.degree) #dict(zip(G.nodes(),df_centralities['degs']))
d_clos = dict(zip(G.nodes(), df_centralities['clos']))
d_betw = dict(zip(G.nodes(), df_centralities['betw']))
d_eigen = dict(zip(G.nodes(), df_centralities['eigen']))
d_centralities = dict(zip(l_sym,zip(d_deghubs.values(),d_clos.values(),d_betw.values(),d_eigen.values())))

l_features = []
for i in d_centralities.items():
    k=list(i)
    l_features.append(k)


#####################
# precalulated Layouts 
#####################

portrait2D = 'input/2D_'+netlayout+'_layout_'+organism
portrait3D = 'input/3D_'+netlayout+'_layout_'+organism
topographic ='input/topographic_'+netlayout+'_layout_'+organism
geodesic = 'input/geodesic_'+netlayout+'_layout_'+organism

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
df2D = pd.read_csv(portrait2D+'.csv', header=None)
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
df3D = pd.read_csv(portrait3D+'.csv', header=None)
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
dfland = pd.read_csv(topographic+'.csv', header=None)
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
dfsphere = pd.read_csv(geodesic+'.csv', header=None)
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
                    #html.Img(src="assets/neon_logo_nobrackets.png",style = {'display':'inline', 'height':'100px'}),
                    #html.H1(" | A Framework for Interpretable Network Visualizations", style={'margin-left':'20px','margin-top':'38px'}),
                    html.H1("cartoGRAPHs", style={'margin-left':'20px','margin-top':'38px'}),
                    html.H1(" | A Framework for Interpretable Network Visualizations", style={'margin-left':'20px','margin-top':'38px'}),
                    ],
                    style = {'display':'inline-flex'},
                ),
                html.Div(className = 'nine columns', 
                children = [
                    # Network picture
                    html.Div(
                        id='layout-graph'
                        ),
                    html.P('This visualization app is currently under construction. Feel free to get in touch for bug reports, comments, suggestions : chris@menchelab.com'),
                ]),
                html.Div(className = 'three columns', 
                children = [
                    html.H6('ORGANISM'),
                    html.Div(children=[
                        dcc.Dropdown(
                            id='dropdown-organism',
                            options=[
                                {'label': 'human', 'value': 'human'},
                                {'label': 'yeast', 'value': 'yeast'},
                            ],
                            placeholder="Select an organism", 
                            style={'color':'#000000'} #font color for dropdown menu
                            )
                    ]),

                    html.Br(),
                    html.Br(),

                    html.H6('NETWORK LAYOUT'),
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
                    html.Br(),
                    html.Br(),
                    
                    html.H6('NETWORK MAP'),
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
                    html.Br(),
                    html.Br(),
                    html.H6('VISUAL PROPERTIES'),
                    html.P('Choose one of the following visualization options, e.g. node colors and size parameters.'),
                    html.Br(),
                    html.P('Node Colour'),
                    html.Div(children=[
                        dcc.Dropdown(
                            options=[
                                #{'label': 'Degree Centrality', 'value': 'deg'},
                                {'label': 'Closeness Centrality', 'value': 'clos'},
                                #{'label': 'Biological Functions', 'value': 'biofunc'},
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
                                #{'label': 'Specific Gene List', 'value': 'genelist'},
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


# call back for network layout 
                        



# Network Layout Map
@app.callback(Output('layout-graph', 'children'),
              [Input('dropdown-map-type', 'value')]
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
    app.run_server(debug=True)