
#print('CSDEBUG: got to app.py')

serving = 'pythonanywhere' # 'lem'



from dash_core_components.express import send_string
from sklearn.metrics.pairwise import _euclidean_distances_upcast


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
elif serving == 'lem':  # lem menchelab
   filePre = '/var/www/cartoGRAPHs_app/cartoGRAPHs_app/'
   
elif serving == 'pythonanywhere': # for pythonanywhere
   filePre = ''
   #print('CSDEBUG: __init turned on asimov flag')


##################################################################################
#
# Initialise the app
myServer = Flask(__name__)
app = dash.Dash(__name__, server=myServer, 
                external_stylesheets=[dbc.themes.BOOTSTRAP], 
                title="cartoGRAPHs",
                #prevent_initial_callbacks=True) 
                #suppress_callback_exceptions=True
                )

#app = dash.Dash()
# in order to work on shinyproxy
# see https://support.openanalytics.eu/t/what-is-the-best-way-of-delivering-static-assets-to-the-client-for-custom-apps/363/5
#app.config.suppress_callback_exceptions = True

# try:
#     app.config.update({
#         'routes_pathname_prefix': os.environ['SHINYPROXY_PUBLIC_PATH'],
#         'requests_pathname_prefix': os.environ['SHINYPROXY_PUBLIC_PATH']
#         })
# except:
    #print('no shinyproxy environment variables')
    #print(' ')

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

slash = '/'
modelnetwork = 'input/model_network_n100.txt'
ppi_elist = 'input/ppi_elist.txt'
ppi_3Dglobal = 'input/3D_global_layout.csv'

dimred = 'umap'
#print('DEBUG: get current working directory:', os.getcwd())
    

######################################
######################################
#         A P P   L A Y O U T 
######################################
######################################
app.layout = html.Div(
            className='app__container',
            id='app__container', children=[
                html.Div(
                id='app__banner',
                children=[
                    html.Div(className="app__banner",
                        children=[
                            html.Img(src=app.get_asset_url('cartoGRAPHs_logo_long2.png'),style={'height':'70px'}),
                            ],
                        )
                    ]),

                ######################################
                #
                # M O D A L at beginning 
                #
                ######################################
                dbc.Modal([
                    dbc.ModalHeader("WELCOME TO cartoGRAPHs"),
                    dbc.ModalBody(
                                "This application can generate 2D and 3D network layouts for interactive network exploration. "
                                "It provides different file downloads of a network layout. Please upload a graph edgelist and select different layout option via dropdown. "
                                "To draw a layout the respective button shall be triggered. "
                                "This application is under development - issues can be raised here: https://github.com/menchelab/cartoGRAPHs_app"
                                ),
                    dbc.ModalFooter(dbc.Button('Close', id='close',className='ml-auto'))
                    ],
                is_open=True,
                id="app__modal",
                centered=True,
                backdrop=True
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
                        html.H6(' 1 | INPUT DATA'),
                        html.P('Upload edge list and optionally features of nodes of the input network.'),
                        
                        # Human Interactome Button 
                        #-------------
                        #html.Button('HUMAN INTERACTOME', id='button-ppi-update', n_clicks_timestamp=0, n_clicks = 0,
                        #    style={'text-align': 'center','width': '100%','margin-top': '5px'}),

                        # Upload Area 
                        #-------------
                        dcc.Upload(className='app__uploads',
                                id='upload-data',
                                last_modified = 0,
                                children=html.Div([
                                    html.A('UPLOAD | EDGELIST'),
                                ]),
                                multiple=False, # Allow multiple files to be uploaded
                                #loading_state# add file restriction 
                            ),
                            dbc.Modal(
                                dbc.ModalBody(
                                "File upload successful!"),
                                id="alert-input",
                                is_open=False,
                                centered=True,
                                #backdrop=True,
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
                        html.Button('DRAW LAYOUT',id='button-graph-update', n_clicks = 0,
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
                                download="interactive_visualization.html"
                        ),


                        html.A(
                                id="download-csv",
                                href="",
                                children=[html.Button('TABLE | csv', id='button-csv', n_clicks=0 ,
                                   style={'text-align': 'center', 'width': '100%', 'margin-top': '5px'}),
                                    ],
                                download="datatable_for_VRnetzer.csv"
                        ),

                        html.A(
                               #id="download-obj",
                               #href="",
                               children=[
                               html.Button('3DModel | obj', id='button-obj', n_clicks=0 ,
                                  style={'text-align': 'center', 'width': '100%', 'margin-top': '5px'}),
                                  dcc.Download(id="download-obj")
                                  ],
                               #download="meshlike_object.obj"
                        ),
                        
                        html.A(
                               #id="download-cyto",
                               #href="",
                               children=[html.Button('Cytoscape | xgmml', id='button-cyto', n_clicks=0 ,
                                  style={'text-align': 'center', 'width': '100%', 'margin-top': '5px'}),
                                  dcc.Download(id="download-cyto")
                               ],
                               #download="cytoscape_graph.xgmml"
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
# Modal opening at start of page
#----------------------------------------
@app.callback(
    Output("app__modal", "is_open"),
    [Input('close', 'n_clicks')],
    [State("app__modal", "is_open")],
)
def toggle_alert(n, is_open):
    if n:
        return not is_open
    return is_open

#----------------------------------------
# Upload input alert 
#----------------------------------------
@app.callback(
    Output("alert-input", "is_open"),
    [Input('upload-data', 'contents')],
    [State("alert-input", "is_open")],
)
def toggle_alert(n, is_open):
    if n:
        return not is_open
    return is_open


#----------------------------------------
# Network Layouts + Maps
#----------------------------------------
@app.callback(
            [Output('layout-graph-figure', 'figure'),
             Output('layout-graph-table', 'data'),
             #Output('layout-cyto', 'data')
             ],

            ######################
            #
            #     I N P U T S
            #
            ######################
            # button "DRAW LAYOUT"
              [Input('button-graph-update','n_clicks'),

            # button for STARTING NETWORK
              #Input('button-ppi-update', 'n_clicks')
              ],

            # WINDOW for upload CONTENT
              [State('upload-data', 'contents'),
            
            # WINDOW for upload FILENAME
               State('upload-data', 'filename'),
              
            # layout type
               State('dropdown-layout-type','value'),

            # map type
               State('dropdown-map-type', 'value'),
            
            # nodesize input 
               State ('nodesize-slider', 'value'),
            
            # link size input 
               State('linksize-slider', 'value'),
            
            # link transparency input 
               State('linkstransp-slider', 'value')]
            
            , prevent_initial_call=False
            )

def update_graph(
                # INPUTS
                buttondrawclicks, # 1 : 'button-graph-update'
                #buttonnetworkclicks, # 4 : button of start network 

                # STATES
                inputcontent, # 2 : input file content
                inputfile, # 3 : input file name
                layoutvalue, # 5 : for network layout 
                mapvalue, # 6 : for network map category
                nodesizevalue, # 7 :'nodesize-slider'
                linksizevalue, # 8 : 'linksize-slider'
                linkstranspvalue, # 9 : 'linktransparency-slider'
                ):

                #---------------------------------------
                # very start of app
                #---------------------------------------     
                if buttondrawclicks == 0:
                        #print('enter network display - very start')
                        #G = nx.read_edgelist(filePre + modelnetwork)
                        
                        # for pythonanywhere:
                        G = nx.read_edgelist(os.path.join(server.root_path,modelnetwork))
                        
                        posG, colours, l_feat = portrait3D_global(G,dimred)  

                        # for datatable of VRNetzer
                        namespace='global3d'
                        df_vrnetzer = export_to_csv3D_app(namespace,posG,colours)
                        dict_vrnetzer = [df_vrnetzer.to_dict()]

                        # for figure html download
                        fig3D_global = draw_layout_3D(G,posG, l_feat, colours, nodesizevalue, 1-linkstranspvalue, linksizevalue) 

                        return fig3D_global, dict_vrnetzer


                        # read human ppi layout from VRnetzer format 
                        #G = nx.read_edgelist(filePre + ppi_elist)
                        #fig3D_start,df_vrnetzer = import_vrnetzer_csv(G, filePre + ppi_3Dglobal)
                        #dict_vrnetzer = [df_vrnetzer.to_dict()]
                        #return fig3D_start, dict_vrnetzer
                
                elif inputcontent is None: 
                        # for pythonanywhere:
                        G = nx.read_edgelist(os.path.join(server.root_path,modelnetwork)) 
                        
                        #G = nx.read_edgelist(filePre + modelnetwork) #(os.path.join(serverPath),modelnetwork)

                #---------------------------------------
                # toggle between layouts selected via dropdowns
                #---------------------------------------     
                else:
                        G = parse_Graph(inputcontent,inputfile)

            
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
                                    posG, colours, l_feat = portrait2D_local(G, dimred) #include a button for 'tsne' or 'umap'

                                    namespace='local2d'
                                    df_vrnetzer = export_to_csv2D_app(namespace,posG,colours)
                                    dict_vrnetzer = [df_vrnetzer.to_dict()]
                                    
                                    # for figure html download
                                    fig2D = draw_layout_2D(G, posG, l_feat, colours, nodesizevalue, 1-linkstranspvalue, linksizevalue) 

                                    return fig2D, dict_vrnetzer

                            elif layoutvalue == 'global':
                                    posG, colours, l_feat = portrait2D_global(G, dimred)

                                    namespace='global2d'
                                    df_vrnetzer = export_to_csv2D_app(namespace,posG,colours)
                                    dict_vrnetzer = [df_vrnetzer.to_dict()]
                                    
                                    # for figure html download
                                    fig2D = draw_layout_2D(G, posG, l_feat, colours, nodesizevalue, 1-linkstranspvalue, linksizevalue) 

                                    return fig2D, dict_vrnetzer

                            elif layoutvalue == 'importance':
                                    posG, colours, l_feat = portrait2D_importance(G, dimred)

                                    namespace='imp2d'
                                    df_vrnetzer = export_to_csv2D_app(namespace,posG,colours)
                                    dict_vrnetzer = [df_vrnetzer.to_dict()]

                                    # for figure html download
                                    fig2D = draw_layout_2D(G, posG, l_feat, colours, nodesizevalue, 1-linkstranspvalue, linksizevalue) 

                                    return fig2D, dict_vrnetzer


                            #elif layoutvalue == 'func':


                ##################
                #
                #  3 D PORTRAIT
                #
                ##################
                elif mapvalue == 'fig3D':
                            if layoutvalue == 'local':
                                    posG, colours, l_feat = portrait3D_local(G, dimred)

                                    namespace='local3d'
                                    df_vrnetzer = export_to_csv3D_app(namespace,posG,colours)
                                    dict_vrnetzer = [df_vrnetzer.to_dict()]

                                    # for figure html download
                                    fig3D = draw_layout_3D(G,posG, l_feat, colours, nodesizevalue, 1-linkstranspvalue, linksizevalue) 

                                    return fig3D, dict_vrnetzer

                            elif layoutvalue == 'global':
                                    posG, colours, l_feat = portrait3D_global(G,dimred)  

                                    # for datatable of VRNetzer
                                    namespace='global3d'
                                    df_vrnetzer = export_to_csv3D_app(namespace,posG,colours)
                                    dict_vrnetzer = [df_vrnetzer.to_dict()]

                                    # for figure html download
                                    fig3D = draw_layout_3D(G,posG, l_feat, colours, nodesizevalue, 1-linkstranspvalue, linksizevalue) 

                                    return fig3D, dict_vrnetzer
                                

                            elif layoutvalue == 'importance':
                                    posG, colours, l_feat = portrait3D_importance(G, dimred)

                                    namespace='imp3d'
                                    df_vrnetzer = export_to_csv3D_app(namespace,posG,colours)
                                    dict_vrnetzer = [df_vrnetzer.to_dict()]
                                    
                                    # for figure html download
                                    fig3D = draw_layout_3D(G,posG, l_feat, colours, nodesizevalue, 1-linkstranspvalue, linksizevalue) 

                                    return fig3D, dict_vrnetzer


                            #elif layoutvalue == 'functional':


                        
                ##################
                #
                #  TOPOGRAPHIC
                #
                ##################
                elif mapvalue == 'figland':
                            deg = dict(G.degree())
                            z_list = list(deg.values()) # U P L O A D L I S T  with values if length G.nodes !!!

                            if layoutvalue == 'local':
                                    posG, colours, l_feat = topographic_local(G,z_list, dimred)

                                    namespace='localtopo'
                                    df_vrnetzer = export_to_csv3D_app(namespace,posG,colours)
                                    dict_vrnetzer = [df_vrnetzer.to_dict()]
                                    
                                    # for figure html download
                                    fig3D = draw_layout_3D(G,posG, l_feat, colours, nodesizevalue, 1-linkstranspvalue, linksizevalue) 
                                    
                                    return fig3D , dict_vrnetzer
                                
                            elif layoutvalue == 'global':
                                    posG, colours, l_feat  = topographic_global(G,z_list, dimred)

                                    namespace='globaltopo'
                                    df_vrnetzer = export_to_csv3D_app(namespace,posG,colours)
                                    dict_vrnetzer = [df_vrnetzer.to_dict()]
 
                                    # for figure html download
                                    fig3D = draw_layout_3D(G,posG, l_feat, colours, nodesizevalue, 1-linkstranspvalue, linksizevalue) 
                                    
                                    return fig3D,dict_vrnetzer

                            elif layoutvalue == 'importance':

                                    closeness = nx.closeness_centrality(G)
                                    d_clos_unsort  = {}
                                    for node, cl in sorted(closeness.items(), key = lambda x: x[1], reverse = 0):
                                        d_clos_unsort [node] = round(cl,4)
                                    d_clos = {key:d_clos_unsort[key] for key in G.nodes()}
                                    z_list = list(d_clos.values())

                                    posG, colours, l_feat = topographic_importance(G, z_list, dimred)

                                    namespace='imptopo'
                                    df_vrnetzer = export_to_csv3D_app(namespace,posG,colours)
                                    dict_vrnetzer = [df_vrnetzer.to_dict()]

                                    # for figure html download
                                    fig3D = draw_layout_3D(G,posG, l_feat, colours, nodesizevalue, 1-linkstranspvalue, linksizevalue) 
                                    
                                    return fig3D, dict_vrnetzer


                            #elif layoutvalue == 'functional':


                ##################
                #
                #  GEODESIC
                #
                ##################
                elif mapvalue == 'figsphere':
                                radius = dict(G.degree()) # U P L O A D L I S T  with values if length G.nodes !!!

                                if layoutvalue == 'local':
                                    posG, colours, l_feat = geodesic_local(G,radius)

                                    namespace='localgeo'
                                    df_vrnetzer = export_to_csv3D_app(namespace,posG,colours)
                                    dict_vrnetzer = [df_vrnetzer.to_dict()]

                                    # for figure html download
                                    fig3D = draw_layout_3D(G,posG, l_feat, colours, nodesizevalue, 1-linkstranspvalue, linksizevalue) 
                                    
                                    return fig3D, dict_vrnetzer


                                elif layoutvalue == 'global':
                                    posG, colours, l_feat = geodesic_global(G,radius)

                                    namespace='localgeo'
                                    df_vrnetzer = export_to_csv3D_app(namespace,posG,colours)
                                    dict_vrnetzer = [df_vrnetzer.to_dict()]

                                    # for figure html download
                                    fig3D = draw_layout_3D(G,posG, l_feat, colours, nodesizevalue, 1-linkstranspvalue, linksizevalue) 
                                    
                                    return fig3D, dict_vrnetzer


                                elif layoutvalue == 'importance':
                                    posG, colours, l_feat = geodesic_importance(G,radius)

                                    namespace='localgeo'
                                    df_vrnetzer = export_to_csv3D_app(namespace,posG,colours)
                                    dict_vrnetzer = [df_vrnetzer.to_dict()]

                                    # for figure html download
                                    fig3D = draw_layout_3D(G,posG, l_feat, colours, nodesizevalue, 1-linkstranspvalue, linksizevalue) 
                                    
                                    return fig3D, dict_vrnetzer


                                #elif layoutvalue == 'functional':




#------------------------------------
# DOWNLOAD FIGURE
#------------------------------------
@app.callback(Output('download-figure', 'href'),
            [Input('button-figure', 'n_clicks')],
            [Input('layout-graph-figure','figure')]
            , prevent_initial_callback=True
            )

def get_image(n_clicks,
        figure):
        #print('CDEBUG: get_image')
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
    return dcc.send_file(filePre + 'output/download_figure.html',
                     mimetype='text:html',
                     as_attachment=True)


#----------------------------------------
# DOWNLOAD CSV
#----------------------------------------
@app.callback(
    Output('download-csv', 'href'),
    [Input('button-csv', 'n_clicks')],
    [Input('layout-graph-table','data')]
    , prevent_initial_callback=True
    )

def get_table(n_clicks,
        table):
        #print('CDEBUG: get_table')
        for i in table:
                df = pd.DataFrame(i)
                #df = pd.DataFrame.from_dict(table, orient='index')
                #print(df)
                df_csv = df.to_csv(index=True,header=False, encoding='utf-8')
                csv_string = "data:text/csv;charset=utf-8," + urlquote(df_csv)
                #csv_string = filePre + "data:text/csv;charset=utf-8," + urlquote(csv_string)

                return csv_string

@myServer.route(filePre + "/download/urlToDownload")
def download_table():
    return dcc.send_data_frame(filePre + 'output/download_table.csv',
                     mimetype='text:csv',
                     as_attachment=True
                     )


#----------------------------------------
# DOWNLOAD OBJ
#----------------------------------------
@app.callback(
    Output('download-obj', 'data'),
    [Input('button-obj', 'n_clicks')],
    [State('layout-graph-table','data')]
    , prevent_initial_callback=True
    )
def get_obj(n_clicks, data):
    if data is not None:
        for i in data:
            df = pd.DataFrame(i)            
            df.columns = ['x','y','z','r','g','b','a','namespace']
            ids = [str(i) for i in list(df.index)]
            x = list(df['x'])
            y = list(df['y'])
            z = list(df['z'])
            posG = dict(zip(ids,zip(x,y,z)))
            myfile = to_obj(posG)

            obj_string = ""
            for ele in myfile:
                obj_string += ele 
            
            return dict(content=obj_string,filename='mesh.obj')
    else:
        pass

@myServer.route(filePre + "/download/urlToDownload")
def download_obj():
    return dcc.send_file('download_object.obj',
                     mimetype='text:plain',
                     as_attachment=True)

#----------------------------------------
# DOWNLOAD for Cytoscape 
#----------------------------------------
@app.callback(
    Output('download-cyto', 'data'),
    [Input('button-cyto', 'n_clicks')],
    [State('layout-graph-table','data')]
    , prevent_initial_callback=True
    )
def get_xgmml(n_clicks, data):
    if data is not None:
        for i in data:
            df = pd.DataFrame(i)            
            df.columns = ['x','y','z','r','g','b','a','namespace']
            ids = [str(i) for i in list(df.index)]
            x = list(df['x'])
            y = list(df['y'])
            z = list(df['z'])
            posG = dict(zip(ids,zip(x,y,z)))
            newG = nx.Graph()
            newG.add_nodes_from(posG.keys())

            for node,coords in posG.items():
                newG.nodes[node]['pos']= coords

            fi = filePre + 'interoperable_graphfile.xgmml'
            #with open (fi,'w') as f:
            graphfile = graph_to_xgmml(fi, newG, 'graph')
            graph_string = ""
            for ele in graphfile:
                graph_string += ele 
        
            return dict(content=graph_string,filename='interoperable_graphfile.xgmml')
    else:
        pass

@myServer.route(filePre + "/download/urlToDownload")
def download_xgmml():
    return dcc.send_file('interoperable_graphfile.xgmml',
                     as_attachment=True)


# --------------------------------------------------------------------------------------------------------------------------

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
