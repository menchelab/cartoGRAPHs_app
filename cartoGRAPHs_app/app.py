print('CSDEBUG: got to app.py')

try:
    print('CSDEBUG: attempting app_main import, in try')
    from app_main import *
    print('CSDEBUG: app_main import * FROM app.py')
except:
    print('CSDEBUG: attempting app_main import, in except')
    from .app_main import *
    print('CSDEBUG: .app_main import * FROM app.py')



# toggle for if from asimov or local
if __name__ == '__main__':
    filePre = ''
    print('CSDEBUG: __init turned on local flag')
else:  # asimov
    filePre = '/var/www/cartoGRAPHs_app/cartoGRAPHs_app/'
    print('CSDEBUG: __init turned on asimov flag')



##################################################################################
#
# Initialise the app
myServer = Flask(__name__)
app = dash.Dash(__name__, server=myServer)#,
                #title="cartoGRAPHs",)
                #prevent_initial_callbacks=True) #,suppress_callback_exceptions=True)

@myServer.route('/favicon.ico')
def favicon():
    return flask.send_from_directory(os.path.join(server.root_path, 'assets'),
                                     'favicon.ico')
#
##################################################################################

print('CSDEBUG: myServer run from app.py')
print('sample path: ' + filePre + 'assets/cartoGraphs_logo_long2.png')
print('get_asset_url: ' + app.get_asset_url('cartoGraphs_logo_long2.png'))

##################################################################################
##################################################################################
#
#                              cartoGRAPHs A P P
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
                    html.Img(src=app.get_asset_url('cartoGraphs_logo_long2.png'),style={'height':'70px'}),
                    #html.Img(src=filePre + 'assets/cartoGraphs_logo_long2.png',style={'height':'70px'}),
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
                        html.P('Upload edge list or choose model network.'),
                        dcc.Upload(
                                id='upload-data',
                                children=html.Div([
                                    html.A('Upload an edgelist here.', style={'text-decoration':'none','font-weight': '300'}),
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
                        download="plotly_graph.html"
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
                        html.P('Download Layouts of the Human Protein-Protein Interaction Network.'),

                        #html.Button('2D PORTRAIT', id='button-ppi-2d', n_clicks=0 ,
                        #            style={'text-align': 'center',
                        #            'width': '100%', 'margin-top': '5px', #'margin-right':'2px',#'display':'inline-block',
                        #           }),
                        #dcc.Download(id='download-ppi2d'),


                        html.A(
                                id="download-ppi3d",
                                href="",
                                children=[html.Button('3D PORTRAIT | PPI', id='button-ppi3d', n_clicks=0,
                                   style={'text-align': 'center', 'width': '100%', 'margin-top': '5px'}),
                                    ],
                        download="plotly_graph_ppi3d.html"
                        ),

                        html.A(
                                id="download-ppitopo",
                                href="",
                                children=[html.Button('TOPOGRAPHIC MAP | PPI', id='button-ppitopo', n_clicks=0,
                                   style={'text-align': 'center', 'width': '100%', 'margin-top': '5px'}),
                                    ],
                        download="plotly_graph_ppitopo.html"
                        ),


                        html.A(
                                id="download-ppigeo",
                                href="",
                                children=[html.Button('GEODESIC MAP | PPI', id='button-ppigeo', n_clicks=0,
                                   style={'text-align': 'center', 'width': '100%', 'margin-top': '5px'}),
                                    ],
                        download="plotly_graph_ppigeo.html"
                        ),

                    ]),

                html.Div(className = 'footer',
                    children=[
                        html.P('This visualization app is frequently updated. We are happy to receive comments and suggestions via Github/menchelab/cartoGRAPHs'),
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
            csv_string = filePre + "data:text/csv;charset=utf-8," + urlquote(csv_string)
            return csv_string

@myServer.route(filePre + "/download/urlToDownload")
def download_table():
    return dcc.send_dataframe(filePre + 'output/download_figure.csv',
                     mimetype='text:csv',
                     attachment_filename='downloadFile.csv',
                     as_attachment=True
                     )

#------------------------------------
# DOWNLOAD FIGURE
#------------------------------------
# @app.callback(Output('download-figure', 'href'),
#             [Input('button-figure', 'n_clicks')],
#             [Input('layout-graph-figure','figure')]
#             )
# def get_image(n_clicks,figure):
#     #if n_clicks:
#         buffer = io.StringIO()
#         plotly.io.write_html(figure,buffer)
#         html_bytes = buffer.getvalue().encode()
#         encoded = b64encode(html_bytes).decode()
#         string = "data:text/html;base64," + encoded
#         return string
#
# @myServer.route("/download/urlToDownload")
# def download_figure():
#     return dcc.send_file('output/download_figure.html',
#                      mimetype='text:html',
#                      attachment_filename='downloadFile.html',
#                      as_attachment=True)

#------------------------------------
# PPI / Figures Manuscript
#------------------------------------

# 3D PORTRAIT PPI - Thesis
#___________________________
@app.callback(Output('download-ppi3d', 'href'),
            Input('button-ppi3d', 'n_clicks'),
            #prevent_initial_call=True
            )
def get_ppi(n_clicks):
    if n_clicks != 0:
        G_ppi = nx.read_edgelist(filePre + 'input/ppi_elist.txt')
        port3d_ppi = filePre + 'input/3D_global_layout_human.csv'
        ppi_portrait3d = import_vrnetzer_csv(G_ppi,port3d_ppi)

        buffer = io.StringIO()
        plotly.io.write_html(ppi_portrait3d,buffer)
        html_bytes = buffer.getvalue().encode()
        encoded = b64encode(html_bytes).decode()
        string = "data:text/html;base64," + encoded
        return string
    #else:
    #    raise PreventUpdate

@myServer.route(filePre + "/download/urlToDownload")
def download_figure_ppi3d():
    return dcc.send_file(filePre + 'output/download_ppi-3d.html',
                     mimetype='text:html',
                     attachment_filename='downloadFile_ppi-3d.html',
                     as_attachment=True)


# 2D PORTRAIT PPI - functional Diseases / Manuscript
#_________________________________________________________



# 2D PORTRAIT PPI - essentiality / Manuscript
#_________________________________________________________



# TOPOGRAPHIC MAP - Disease landscape / Manuscript
#_________________________________________________________
@app.callback(Output('download-ppitopo', 'href'),
            Input('button-ppitopo', 'n_clicks'),
            #prevent_initial_call=True
            )
def get_ppi(n_clicks):
    if n_clicks != 0:
        G_ppi = nx.read_edgelist(filePre + 'input/ppi_elist.txt')
        topo_ppi = filePre + 'input/topographic_global_layout_human.csv'
        ppi_topo = import_vrnetzer_csv(G_ppi,topo_ppi)
        buffer = io.StringIO()

        plotly.io.write_html(ppi_topo,buffer)
        html_bytes = buffer.getvalue().encode()
        encoded = b64encode(html_bytes).decode()
        string = filePre + "data:text/html;base64," + encoded
        return string
    #else:
    #    raise PreventUpdate


@myServer.route("/download/urlToDownload")
def download_figure_ppitopo():
    return dcc.send_file(filePre + 'output/download_ppi-topo.html',
                     mimetype='text:html',
                     attachment_filename='downloadFile_ppi-topo.html',
                     as_attachment=True)


# GEODESIC MAP - Rare disease variants/seeds / Manuscript
#_________________________________________________________
@app.callback(Output('download-ppigeo', 'href'),
            [Input('button-ppigeo', 'n_clicks')],
            #prevent_initial_call=True
            )
def get_ppi(n_clicks):
    if n_clicks != 0:
        G_ppi = nx.read_edgelist(filePre + 'input/ppi_elist.txt')
        geo_ppi = filePre + 'input/geodesic_global_layout_human.csv'
        ppi_topo = import_vrnetzer_csv(G_ppi,geo_ppi)
        buffer = io.StringIO()

        plotly.io.write_html(ppi_topo,buffer)
        html_bytes = buffer.getvalue().encode()
        encoded = b64encode(html_bytes).decode()
        string = filePre + "data:text/html;base64," + encoded
        return string
        #raise PreventUpdate
    #else:
    #    raise PreventUpdate

@myServer.route("/download/urlToDownload")
def download_figure_ppigeo():
    return dcc.send_file(filePre + 'output/download_ppi-geo.html',
                     mimetype='text:html',
                     attachment_filename='downloadFile_ppi-geo.html',
                     as_attachment=True)



#----------------------------------------
# Network Layout + Map
#----------------------------------------
@app.callback(

            [Output('layout-graph-figure', 'figure'),
             Output('layout-graph-table', 'data')],

            # button "DRAW LAYOUT"
              [Input('button-graph-update','n_clicks')],

            # INPUT WINDOW for upload CONTENT
              Input('upload-data', 'contents'),

            # button "MODEL NETWORK" for network input
              [Input('button-network-type', 'n_clicks')],

            # INPUT WINDOW for upload FILENAME
              Input('upload-data', 'filename'),
              Input('upload-data','last_modified'),

            # states of layout and map
              [State('dropdown-layout-type','value')],
              [State('dropdown-map-type', 'value')],
              )

def update_graph(buttonclicks, #'button-graph-update'
                inputcontent, #'upload-data' content
                modelclicks, #'button-network-type'
                inputfile, #'upload-data' filename
                input_lastmod,
                layoutvalue,
                mapvalue):

                #---------------------------------------
                # very start of app
                #---------------------------------------
                if buttonclicks == 0:
                            G = nx.read_edgelist(filePre + 'input/model_network_n1000.txt')
                            fig3D_start,posG,colours = portrait3D_global(G)

                            namespace='exemplarygraph'
                            df_vrnetzer = export_to_csv3D_app(namespace,posG,colours)
                            dict_vrnetzer = [df_vrnetzer.to_dict()]

                            return fig3D_start, dict_vrnetzer

                #---------------------------------------
                # toggle inbetween user input (Network)
                #---------------------------------------

                #if buttonclicks:

                    #---------------------------------------
                    # Model Graph
                    #---------------------------------------
                if buttonclicks or modelclicks:

                    if inputfile is None:
                        G = nx.read_edgelist(filePre + 'input/model_network_n1000.txt')

                    #elif modelclicks:
                    #    G = nx.read_edgelist('input/model_network_n1000.txt')

                    #elif int(modelclicks) > int(input_lastmod):
                    #        G = nx.read_edgelist('input/model_network_n1000.txt')

                    #---------------------------------------
                    # Model Graph
                    #---------------------------------------
                    elif modelclicks:
                            G = nx.read_edgelist(filePre + 'input/model_network_n1000.txt')

                    #---------------------------------------
                    # Upload / Input Graph
                    #---------------------------------------
                    elif inputfile:
                        G = parse_Graph(inputcontent,inputfile)





                    #if buttonclicks:

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

                                del inputfile


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
    app.run_server(debug=False)





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
