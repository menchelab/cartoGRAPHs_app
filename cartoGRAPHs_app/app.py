#print('CSDEBUG: got to app.py')

try:
   #print('CSDEBUG: attempting app_main import, in try')
   from app_main import *
   #print('CSDEBUG: app_main import * FROM app.py')
except:
   #print('CSDEBUG: attempting app_main import, in except')
   from .app_main import *
   #print('CSDEBUG: .app_main import * FROM app.py')

if __name__ == '__main__':
   filePre = ''
   #print('CSDEBUG: __init turned on local flag')
else:  # asimov
   filePre = '/var/www/cartoGRAPHs_app/cartoGRAPHs_app/'
   #filePre = ''
   #print('CSDEBUG: __init turned on asimov flag')


##################################################################################
#
# Initialise the app
myServer = Flask(__name__)
app = dash.Dash(__name__, server=myServer) #,external_stylesheets=[dbc.themes.BOOTSTRAP])#
                #title="cartoGRAPHs")
                #prevent_initial_callbacks=True) #,suppress_callback_exceptions=True)

#app = dash.Dash()
# in order to work on shinyproxy
# see https://support.openanalytics.eu/t/what-is-the-best-way-of-delivering-static-assets-to-the-client-for-custom-apps/363/5
app.config.suppress_callback_exceptions = True
try:
    app.config.update({
        'routes_pathname_prefix': os.environ['SHINYPROXY_PUBLIC_PATH'],
        'requests_pathname_prefix': os.environ['SHINYPROXY_PUBLIC_PATH']
        })
except:
    #print('no shinyproxy environment variables')
    print(' ')

@myServer.route('/favicon.ico')
def favicon():
    return flask.send_from_directory(os.path.join(server.root_path, 'assets'),'favicon.ico')
#
##################################################################################

#print('CSDEBUG: myServer run from app.py')

##################################################################################
##################################################################################
#
#                              cartoGRAPHs A P P
#
##################################################################################
##################################################################################

modelnetwork = 'input/model_network_n100.txt'
ppi_elist = 'input/ppi_elist.txt'
ppi_3Dglobal = 'input/3D_global_layout.csv'

dimred = 'umap'
#print('DEBUG: get current working directory:', os.getcwd())


######################################
#
#           MODAL
#
######################################
def modal():
    return dbc.Modal(className='app__modal',
        children=[
            dbc.ModalHeader("WElCOME TO cartoGRAPHs"),
            dbc.ModalBody(
                          "This application can generate 2D and 3D network layouts for interactive network exploration. "
                          "It provides downloads of interactive figures, a table format for a VR analytics platform and many more. "
                          "It is currently under development - issues are very welcome to be raised here: https://github.com/menchelab/CartoGRAPHs_app"
                          )
                ],
        is_open=True,
        id="modal-centered",
        size='m',
        centered=True,
    )


######################################
#
#           BANNER / LOGO
#
######################################
def banner():
    return html.Div(
        id='app__banner',
        children=[
            html.Div(className="app__banner",
                children=[
                    html.Img(src=app.get_asset_url('cartoGRAPHs_logo_long2.png'),style={'height':'70px'}),
                    ],
                ),])

######################################
######################################
#         A P P   L A Y O U T 
######################################
######################################
app.layout = html.Div(
            className='app__container',
            id='app__container', children=[
                
                banner(), # BANNER / LOGO
                html.Br(),
                modal(),

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
                        html.H6(' 1 | INPUT DATA'),
                        html.P('Upload edge list and optionally features of nodes of the input network.'),
                        
                        # Human Interactome Button 
                        #-------------
                        html.Button('HUMAN INTERACTOME', id='button-ppi-update', n_clicks_timestamp=0, n_clicks = 0,
                            style={'text-align': 'center','width': '100%','margin-top': '5px'}),

                        # Upload Area 
                        #-------------
                        dcc.Upload(className='app__uploads',
                                id='upload-data',
                                last_modified = 0,
                                children=html.Div([
                                    html.A('UPLOAD | EDGELIST'),
                                ]),
                                multiple=False, # Allow multiple files to be uploaded
                                # add file restriction 
                            ),
                        #html.Div(id='output-data-upload'),

                        #----------------------------------------
                        # Choose Model Network - Button
                        #----------------------------------------
                        #html.Div(children=[
                        #    html.Button('MODEL NETWORK', id='button-network-type', n_clicks=0 ,
                        #    style={'text-align': 'center','width': '100%','margin-top': '5px'}),
                        #]),

                        dcc.Upload(className='app__uploads',
                                id='upload-matrix',
                                last_modified = 0,
                                children=html.Div([
                                    html.A('UPLOAD | FEATURES'),
                                ]),
                                multiple=False # DO NOT allow multiple files to be uploaded
                            ),
                        #html.Br(),

                        #----------------------------------------
                        # LAYOUTS (local, global, imp, func)
                        #----------------------------------------
                        html.H6(' 2 | NETWORK LAYOUT'),
                        html.P('Choose one of the listed layouts.'),
                        html.Div(children=[
                            dcc.Dropdown(
                            className='app__dropdown',
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
                        html.H6(' 3 | NETWORK MAP CATEGORY'),
                        html.P('Choose one of the listed map categories.'),
                        html.Div(children=[
                            dcc.Dropdown(
                            className='app__dropdown',
                                id='dropdown-map-type',
                                options=[
                                    {'label': '2D Portrait', 'value': 'fig2D'},
                                    {'label': '3D Portrait', 'value': 'fig3D'},
                                    {'label': 'Topographic Map', 'value': 'figland'},
                                    {'label': 'Geodesic Map', 'value': 'figsphere'},
                                ],
                                placeholder="Select a Layout Map.",
                                ),
                        ]),

                        #----------------------------------------
                        # UPDATE NETWORK BUTTON
                        #----------------------------------------
                        html.Button('DRAW LAYOUT',id='button-graph-update', n_clicks_timestamp=0, n_clicks = 0,
                            style={'text-align': 'center','width': '100%','margin-top': '10px'}),

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
                        # VISUAL SETTINGS 
                        #----------------------------------------
                        html.H6(' 4 | VISUAL MODIFICATIONS'),
                        html.P('Change node size, link size and link transparency and refresh by clicking DRAW LAYOUT.'),
                        html.Br(),
                        html.P('Node Size'),
                        html.Div([
                            dcc.Slider(
                                id='nodesize-slider',
                                min=0,
                                max=10,
                                step=0.1,
                                value=5, 

                            ),
                        ], style={'height':'1rem'}),

                        html.Br(),
                        html.P('Link Size'),
                        html.Div([
                            dcc.Slider(
                                id='linksize-slider',
                                min=0,
                                max=10,
                                step=0.1,
                                value=1,

                            ),
                        ], style={'height':'1rem'}),
                        html.Br(),
                        html.P('Link Transparency'),
                        html.Div([
                            dcc.Slider(
                                id='linkstransp-slider',
                                min=0,
                                max=1,
                                step=0.01,
                                value=0.5,
                            ),
                        ], style={'height':'1rem'}),

                        #----------------------------------------
                        # DOWNLOAD Layouts
                        #----------------------------------------
                        html.H6(' 5 | DOWNLOADS'),
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

                        html.A(
                                id="download-obj",
                                href="",
                                children=[html.Button('3DModel | obj', id='button-obj', n_clicks=0 ,
                                   style={'text-align': 'center', 'width': '100%', 'margin-top': '5px'}),
                                   ],
                        ),

                        html.A(
                                id="download-cyto",
                                href="",
                                children=[html.Button('Cytoscape | gml', id='button-cyto', n_clicks=0 ,
                                   style={'text-align': 'center', 'width': '100%', 'margin-top': '5px'}),
                                   ],
                        ),

                        html.Br(),
                        html.Br(),


                html.Div(className = 'footer',
                    children=[
                        html.Div([
                        html.P(['This visualization app is frequently updated. We are happy to receive comments and suggestions via ',
                        html.A('Github/menchelab/cartoGRAPHs_app', href = 'https://github.com/menchelab/cartoGRAPHs_app', target='_blank',style={'text-decoration':'none'}),
                        ]),
                        ]),
                    ]),
            ])
        ])




#########################################
#
#           C A L L   B A C K S 
#
#########################################


#----------------------------------------
# Modal i.e. pop up window at page loading 
#----------------------------------------

@app.callback(
    Output("modal", "is_open"),
    [Input("open", "n_clicks"), Input("close", "n_clicks")],
    [State("modal", "is_open")],
)

def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open



#----------------------------------------
# Network Layouts + Maps
#----------------------------------------
@app.callback(

            [Output('layout-graph-figure', 'figure'),
             Output('layout-graph-table', 'data')],

            ######################
            #
            #     I N P U T S
            #
            ######################
            # 1 button "DRAW LAYOUT"
              [Input('button-graph-update','n_clicks'),

            # 2 INPUT WINDOW for upload CONTENT
              Input('upload-data', 'contents'),
            
            # 3 INPUT WINDOW for upload FILENAME
              Input('upload-data', 'filename'),

            # 4 button for STARTING NETWORK
              Input('button-ppi-update', 'n_clicks'),
              
            # 5 input of layout type
              Input('dropdown-layout-type','value'),

            # 6 input of map type
              Input('dropdown-map-type', 'value'),
              #],


            ######################
            #
            #     S T A T E S
            #
            ######################

            # 1 State of button network 
              #State[('button-ppi-update', 'n_clicks_timestamp'),
            
            # 2 States of layout and map
              #State('dropdown-layout-type','n_clicks_timestamp'),
            
            # 3 States of layout and map
              #State('dropdown-map-type', 'n_clicks_timestamp'),
            
            # 4 nodesize input 
              Input('nodesize-slider', 'value'),
              #State[('nodesize-slider', 'value'),
            
            # 5 link size input 
              Input('linksize-slider', 'value'),
            
            # 6 link transparency input 
              Input('linkstransp-slider', 'value')]
            )

#def calculate_layout()


def update_graph(
                # INPUT
                buttondrawclicks, # 1 : 'button-graph-update'
                inputcontent, # 2 : input file content
                inputfilename, # 3 : input file name
                buttonnetworkclicks, # 4 : button of start network 
                layoutvalue, # 5 : for network layout 
                mapvalue, # 6 : for network map category

                # STATE 

                #buttonnetworkstate, # 1 : state of preloaded network button
                #layoutstate, # 2 : clicks_timestamp for layout type 
                #mapstate, # 3 : clicks_timestamp for map type 
                nodesizevalue, # 4 :'nodesize-slider'
                linksizevalue, # 5 : 'linksize-slider'
                linkstranspvalue, # 6 : 'linktransparency-slider'
                ):


                #---------------------------------------
                # very start of app
                #---------------------------------------
                if buttonnetworkclicks == 0 and buttondrawclicks == 0:

                        print('enter network display > first time')
                        G = nx.read_edgelist(filePre + ppi_elist)
                        fig3D_start,df_vrnetzer = import_vrnetzer_csv(G, filePre + ppi_3Dglobal)
                        dict_vrnetzer = [df_vrnetzer.to_dict()]

                        return fig3D_start, dict_vrnetzer
                
                #---------------------------------------
                # only run when "Draw layout" is pressed after other input
                #---------------------------------------


                elif inputcontent is not None and layoutvalue is not None and mapvalue is not None:
                    if buttondrawclicks:
                        print('enter buttonclicks')
                        #---------------------------------------
                        # Model Graph
                        #---------------------------------------
                        G = parse_Graph(inputcontent,inputfilename)
                        #print('DEBUG: choose INPUT #1')

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
                                    fig2D_local,posG,colours = portrait2D_local(G, dimred) #include a button for 'tsne' or 'umap'

                                    namespace='local2d'
                                    df_vrnetzer = export_to_csv2D_app(namespace,posG,colours)
                                    dict_vrnetzer = [df_vrnetzer.to_dict()]

                                    return fig2D_local, dict_vrnetzer

                            elif layoutvalue == 'global':
                                    fig2D_global,posG,colours = portrait2D_global(G, dimred)

                                    namespace='global2d'
                                    df_vrnetzer = export_to_csv2D_app(namespace,posG,colours)
                                    dict_vrnetzer = [df_vrnetzer.to_dict()]

                                    return fig2D_global,dict_vrnetzer

                            elif layoutvalue == 'importance':
                                    fig2D_imp,posG,colours = portrait2D_importance(G, dimred)

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
                                    fig3D_local,posG,colours = portrait3D_local(G)# , dimred)

                                    namespace='local3d'
                                    df_vrnetzer = export_to_csv3D_app(namespace,posG,colours)
                                    dict_vrnetzer = [df_vrnetzer.to_dict()]

                                    return fig3D_local, dict_vrnetzer

                            elif layoutvalue == 'global':
                                    #fig3D_global,posG,colours = portrait3D_global(G, dimred, nodesizevalue, linksizevalue, 1-linkstranspvalue)

                                    posG, colours, l_feat = portrait3D_global_(G,dimred)  
                                    namespace='global3d'
                                    df_vrnetzer = export_to_csv3D_app(namespace,posG,colours)
                                    #print(df_vrnetzer)
                                    dict_vrnetzer = [df_vrnetzer.to_dict()]

                                    fig3D_global = draw_layout_3D(G,posG, l_feat, colours, nodesizevalue, 1-linkstranspvalue, linksizevalue) 

                                    return fig3D_global, dict_vrnetzer
                                

                            elif layoutvalue == 'importance':
                                    fig3D_imp, posG, colours = portrait3D_importance(G, dimred)

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
                                # 
                                #                                        ])
                        else: 
                            print('prevent update')
                            raise PreventUpdate


#----------------------------------------
# DOWNLOAD CSV
#----------------------------------------
@app.callback(
    Output('download-csv', 'href'),
    [Input('button-csv', 'n_clicks')],
    [Input('layout-graph-table','data')]
    )

def get_table(n_clicks,table):
    #if n_clicks:
            #for i in table:
            #    df = pd.DataFrame(i)
            df = pd.DataFrame(table)
            #print(df)
            csv_string = df.to_csv(index=False,header=False, encoding='utf-8')
            csv_string = "data:text/csv;charset=utf-8," + urlquote(csv_string)
            #csv_string = filePre + "data:text/csv;charset=utf-8," + urlquote(csv_string)

            return csv_string

@myServer.route(filePre + "/download/urlToDownload")
def download_table():
    return dcc.send_data_frame(filePre + 'output/download_figure.csv',
                     mimetype='text:csv',
                     attachment_filename='downloadFile.csv',
                     as_attachment=True
                     )

#------------------------------------
# DOWNLOAD FIGURE
#------------------------------------
@app.callback(Output('download-figure', 'href'),
            #[Input('button-figure', 'n_clicks')],
            [Input('layout-graph-figure','figure')]
            )
def get_image(#n_clicks,
        figure):
    #if n_clicks:
        #print('CSDEBUG: in get_image')
        buffer = io.StringIO()
        plotly.io.write_html(figure,buffer)
        #print('CSDEBUG: in get_image, plotly.io.write_html successful')
        html_bytes = buffer.getvalue().encode()
        encoded = b64encode(html_bytes).decode()
        #print('CSDEBUG: in get_image, b64encode successful')
        string = "data:text/html;base64," + encoded
        return string

@myServer.route("/download/urlToDownload")
def download_figure():
    #print('CSDEBUG: in download_figure')
    return dcc.send_file('output/download_figure.html',
                     mimetype='text:html',
                     attachment_filename='downloadFile.html',
                     as_attachment=True)




server = app.server
if __name__ == '__main__':
    #print('we are in --main__')
    app.run_server(debug=True,
                   use_reloader=False)











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
