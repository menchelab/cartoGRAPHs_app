import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import pandas as pd
import plotly.graph_objs as pgo


# Initialise the app
app = dash.Dash(__name__)



#########################################
#         FILE INPUT+FIGURES            #
#########################################
# read csv from VR compatible layout 

# 2D portrait 
df2D = pd.read_csv('data/portrait2D_VR.csv', header=None)
df2D.columns = ['id','x','y','z','r','g','b','a','namespace']
ids2D = list(df2D['id'])
x_2D = list(df2D['x'])
y_2D = list(df2D['y'])
posG_2D = dict(zip(ids2D,zip(x_2D,y_2D)))

reds = list(df2D['r'])
blues = list(df2D['b'])
greens = list(df2D['g'])
alpha = list(df2D['a'])

colours = []
for i in range(len(reds)):
    colours.append('rgba'+'('+(str(reds[i])+','+str(blues[i])+','+str(greens[i])+str(alpha[i])+')'))

# plot data 
trace2D = pgo.Scatter(x=x_2D,
                    y=y_2D,
                    mode='markers',
                    marker = dict(color=colours,
                                size = 2)
)

fig2D = pgo.Figure()
fig2D.add_trace(trace2D)
fig2D.update_layout(
                    margin=dict(l=0, r=0, t=0, b=0),
                    template='plotly_dark', showlegend=False, width=800, height=800,
                    scene=dict(
                      xaxis_title='',
                      yaxis_title='',
                      xaxis=dict(nticks=0,tickfont=dict(
                            color='black')),
                      yaxis=dict(nticks=0,tickfont=dict(
                            color='black'))
                            ),  
                    )


# 3D portrait
df3D = pd.read_csv('data/portrait3D_VR.csv', header=None)
df3D.columns = ['id','x','y','z','r','g','b','a','namespace']
ids3D = list(df3D['id'])
x_3D = list(df3D['x'])
y_3D = list(df3D['y'])
z_3D = list(df3D['z'])
posG_3D = dict(zip(ids3D,zip(x_3D,y_3D,z_3D)))

reds = list(df3D['r'])
blues = list(df3D['b'])
greens = list(df3D['g'])
alpha = list(df3D['a'])

cols = []
for i in range(len(reds)):
    cols.append('rgb'+'('+(str(reds[i])+','+str(blues[i])+','+str(greens[i])+')')) #+str(alpha[i])+')'))

fig3D = pgo.Figure()
trace3D = pgo.Scatter3d(x=x_3D,
                    y=y_3D,
                    z=z_3D,
                    mode='markers',
                    text = ids3D,
                    marker = dict(color=cols,
                                size = 1.5)
)
fig3D.add_trace(trace3D)
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

# Landscape



# Spherescape 




#########################################
#              APP LAYOUT               #
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
                    html.H1(" | A Framework for Multidimensional Network Visualization", style={'margin-left':'20px','margin-top':'38px'}),
                    ],
                    style = {'display':'inline-flex'},
                ),
                html.Div(className = 'nine columns', 
                children = [
                    # Network picture
                    html.Div(
                        children = [
                            dcc.Graph(
                                config={'displayModeBar':False},
                                style={'position':'relative','height': '80vh', 'width':'100%'},
                                figure=fig3D),
                    ]),
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
                            options=[
                                {'label': '2D Portrait', 'value': 'fig2D'},
                                {'label': '3D Portrait', 'value': 'fig3D'},
                                {'label': 'Landscape', 'value': 'figLand'},
                                {'label': 'Spherescape', 'value': 'figSphere'},
                            ],
                            placeholder="Select a Layout Type",
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
                            )
                    ]),         
                ])
            ])



#########################################
#              CALL BACKS               #
#########################################

#@app.callback(
#    dash.dependencies.Output('app__container', 'children'),
#    [dash.dependencies.Input('layoutType', 'value')])



#@app.callback(Output('tabs-content','children'),
#            [Input('tabs','value')])

# def render_content(tab):
#     if tab == 'tab-1':
#         return html.Div(
#             children=[
#                 html.H3('Structural Parameters'),
#                 html.Div(
#                     children=[
#                         html.P('Some short description.'),
#                         html.P('Here should be parameter buttons, settings,..'),
#                     ]),
#             html.Div(
#                 children = [
#                     # Network picture
#                     dcc.Graph(
#                         style={'height': 1000,'width': 1000, 'display':'inline'},
#                         figure=fig3D)
#                 ]),
#         ])

#     elif tab == 'tab-2':
#         return html.Div([
#             html.H3('Functional Parameters'),
#             html.Div(
#                 children=[html.P('param 1 blablabla'),
#                 html.P('param 2 blablabla'),
#                 ]
#             )
#         ])
#     elif tab == None:
#         return 


# Run the app
server = app.server
if __name__ == '__main__':
    app.run_server(debug=True)