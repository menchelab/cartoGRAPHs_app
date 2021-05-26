import io
import base64
import networkx as nx
import pandas as pd
import pickle
import plotly
import plotly.express as px
import plotly.graph_objs as pgo
import plotly.offline as py
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, iplot
import plotly.io as pio
import seaborn as sns

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_table


########################################################################################
#
# F U N C T I O N S   T O  L O A D  D A T A 
# 
########################################################################################


def load_graph(organism):
    
    if organism == 'yeast':
    
        data = pickle.load( open( "input/BIOGRID-ORGANISM-Saccharomyces_cerevisiae_S288c-3.5.185.mitab.pickle", "rb" ) )

        filter_score = data[
                            #(data['Interaction Types'] == 'psi-mi:"MI:0915"(physical association)') +
                            (data['Interaction Types'] == 'psi-mi:"MI:0407"(direct interaction)') 
                            #&
                            #(data['Taxid Interactor A'] == "taxid:559292") & 
                            #(data['Taxid Interactor B'] == "taxid:559292") 
        ]

        g = nx.from_pandas_edgelist(filter_score, '#ID Interactor A', 'ID Interactor B')
        g.remove_edges_from(nx.selfloop_edges(g)) #remove self loop

        G_cere = g.subgraph(max(nx.connected_components(g), key=len)) # largest connected component (lcc)
        G = G_cere

        return G
    
    elif organism == 'human':
        
        G = nx.read_edgelist('input/ppi_elist.txt',data=False)
        return G    
    
    else: 
        print('Please choose organism by typing "human" or "yeast"')


def load_genesymbols(G,organism):
    '''
    Load prepared symbols of genes.
    Input: 
    - organism = string; choose from 'human' or 'yeast'

    Return dictionary of geneID (keys) and symbols (values).
    '''  
    if organism == 'yeast':
        df_gID_sym = pd.read_csv('input/DF_gene_symbol_yeast.csv', index_col=0)
        gene_sym = list(df_gID_sym['Sym'])
        gene_id = list(df_gID_sym.index)
        d_gene_sym  = dict(list(zip(gene_id, gene_sym)))
        
        return d_gene_sym 
    
    elif organism == 'human':
        df_gene_sym = pd.read_csv('input/DF_gene_symbol_human.csv', index_col=0)
        sym = list(df_gene_sym['0'])
        l_features = []
        for i in sym:
            l_features.append(i[2:-2])
        d_gene_sym = dict(zip(G.nodes(),l_features))
        
        return d_gene_sym 
  
    else: 
        print('Please choose organism by typing "human" or "yeast"')
        
            

def load_centralities(G,organism):
        '''
        Load prepared centralities of genes.
        Input: 
        - G = Graph
        - organism = string; choose from 'human' or 'yeast'

        Return dictionary with genes as keys and four centrality metrics as values.
        '''
        df_centralities = pd.read_csv('input/Features_centralities_Dataframe_'+organism+'.csv', index_col=0)

        d_deghubs = dict(G.degree()) 
        d_clos = dict(zip(G.nodes(), df_centralities['clos']))
        d_betw = dict(zip(G.nodes(), df_centralities['betw']))
        d_eigen = dict(zip(G.nodes(), df_centralities['eigen']))

        d_centralities = dict(zip(list(G.nodes),zip(d_deghubs.values(),d_clos.values(),d_betw.values(),d_eigen.values())))

        #cent_features = []
        #for i in d_centralities.items():
        #    k=list(i)
        #    cent_features.append(k)
        
        return d_centralities



def load_essentiality(G, organism):
        '''
        Load prepared essentiality state of organism. 
        Input: 
        - organism = string; choose from 'human' or 'yeast'

        Return lists of genes, split based on essentiality state. 
        '''
        if organism == 'human':
            
            # ESSENTIALITY 
            # get dataframe with ENSG-ID and essentiality state 
            df_human_ess = pd.read_table("input/human_essentiality.txt", delim_whitespace=True)

            # create dict with ENSG-ID:essentiality state 
            ensg_id = list(set(df_human_ess['sciName']))
            gene_ess = list(df_human_ess['locus'])
            d_ensg_ess = dict(zip(ensg_id, gene_ess))

            # match ENSG-ID with entrezID
            # "engs_to_entrezid": entrezIDs were matched with "ensg_id.txt" via "DAVID Database" (https://david.ncifcrf.gov/conversion.jsp)
            df_human_ensg_entrez = pd.read_table('input/ensg_to_entrezid.txt') # delim_whitespace=False)
            df_human_ensg_entrez.dropna()

            df = df_human_ensg_entrez
            df['To'] = df['To'].fillna(0)
            df['To'] = df['To'].astype(int)
            df_human_ensg_entrez = df

            # create dict with ENGS-ID: entrezID
            ensgid = list(df_human_ensg_entrez['From']) #engs ID
            entrezid = list(df_human_ensg_entrez['To']) #entrez ID 

            # dict with engsid : entrezid
            d_ensg_entrez = dict(zip(ensgid, entrezid))

            # create dict with entrezID:essentiality state 
            d_id_ess_unsorted = {}
            for ens,ent in d_ensg_entrez.items():
                for en, ess in d_ensg_ess.items():
                    if ens == en:
                        d_id_ess_unsorted[str(ent)] = ess


            # check if G.nodes match entrezID in dict and sort according to G.nodes 
            d_gid_ess = {}
            for k,v in d_id_ess_unsorted.items():
                if k in G.nodes():
                    d_gid_ess[k]=v

            # create dict with rest of G.nodes not in dict (entrezID:essentiality)
            d_gid_rest = {}
            for g in G.nodes():
                if g not in d_gid_ess.keys():
                    d_gid_rest[g]='not defined'

            #print(len(d_gid_rest)+len(d_gid_ess)) # this should match G.nodes count 

            # merge both dicts
            d_gid_ess_all_unsorted = {**d_gid_ess, **d_gid_rest}

            # sort -> G.nodes()
            d_gID_all = {key:d_gid_ess_all_unsorted[key] for key in G.nodes()}

            essential_genes = []
            non_ess_genes = []
            notdefined_genes = [] 
            for k,v in d_gID_all.items():
                if v == 'E':
                    essential_genes.append(k)
                elif v == 'NE':
                    non_ess_genes.append(k)
                else:
                    notdefined_genes.append(k)
                    
            return essential_genes,non_ess_genes,notdefined_genes
        
        
        elif organism == 'yeast':
            
            # ESSENTIALITY 
            cere_gene =pd.read_csv("input/Saccharomyces cerevisiae.csv",
                       delimiter= ',',
                       skipinitialspace=True)

            cere_sym = list(cere_gene['symbols'])
            cere_ess = list(cere_gene['essentiality status'])
            cere_sym_essentiality = dict(zip(cere_sym, cere_ess))

            d_cere_ess = {}
            d_cere_noess = {}
            d_cere_unknown = {}

            for node,es in cere_sym_essentiality.items():
                if es == 'E':
                    d_cere_ess[node]=es
                elif es == 'NE':
                    d_cere_noess[node]=es

            d_cere_alless = {}
            for nid, sym in g_ID_sym.items():
                for sy,ess in cere_sym_essentiality.items():
                    if sym == sy:
                        d_cere_alless[nid] = ess

            d_cere_unknown = {} 
            for g in G.nodes():
                if g not in d_cere_alless.keys():
                    d_cere_unknown[g]='status unkonwn'

            d_geneID_ess = {**d_cere_unknown, **d_cere_alless}

            d_gID_ess = {}
            d_gID_noess = {}
            d_gID_notdef = {}

            for k,i in d_geneID_ess.items():
                if i == 'E':
                    d_gID_ess[k] = i
                elif i == 'NE':
                    d_gID_noess[k] = i
                else: 
                    d_gID_notdef[k] = 'not defined'

            d_gID_all_unsorted = {**d_gID_ess, **d_gID_noess, **d_gID_notdef}
            d_gID_all = {key:d_gID_all_unsorted[key] for key in G.nodes()}

            essential_genes = []
            non_ess_genes = []
            notdefined_genes = [] 
            for k,v in d_gID_all.items():
                if v == 'E':
                    essential_genes.append(k)
                elif v == 'NE':
                    non_ess_genes.append(k)
                else:
                    notdefined_genes.append(k)
            
            return essential_genes,non_ess_genes,notdefined_genes

        else:
            print('Please choose organism by typing "human" or "yeast"')

            
            
def load_structural_datamatrix(G,organism,netlayout):
    '''
    Load precalculated Matrix with N genes and M features.
    Input: 
    - path = directory of file location
    - organism = string; choose from 'human' or 'yeast'
    - netlayout = string; choose a network layout e.g. 'local', 'global', 'importance', 'funct-bio', 'funct-cel', 'funct-mol', funct-dis'

    Return Matrix based on choice.
    '''
    path = 'input/'
    
    if netlayout == 'local':
        
        DM_adj = pd.read_csv(path+'Adjacency_Dataframe_'+organism+'.csv', index_col=0)
        DM_adj.index = list(G.nodes())
        DM_adj.columns = list(G.nodes())
        
        return DM_adj
    
    elif netlayout == 'global':
        
        #r = 0.9
        #alpha = 0.9-1.0
        
        DM_m_visprob_transposed = pd.read_csv(path+'RWR_Dataframe_'+organism+'.csv', index_col=0)
        DM_m_visprob_transposed.index = list(G.nodes())
        DM_m_visprob_transposed.columns = list(G.nodes())
        
        return DM_m_visprob_transposed
    
    elif netlayout == 'importance':
        
        df_centralities = load_centralities(organism)
        DM_centralities = pd.DataFrame(distance.squareform(distance.pdist(df_centralities, 'cosine')))

        DM_centralities = round(DM_centralities,6)
        DM_centralities.index = list(G.nodes())
        DM_centralities.columns = list(G.nodes())
        
        return DM_centralities
    
    elif netlayout == 'funct-bio' and organism == 'human':
        
        DM_BP = pd.read_csv(path+'DistanceMatrix_goBP_Dataframe_human_cosine.csv', index_col=0)
        DM_BP_round = DM_BP.round(decimals=6)
        
        return DM_BP_round
    
    
    elif netlayout == 'funct-mol' and organism == 'human':
        
        DM_MF = pd.read_csv('input/DistanceMatrix_goMF_Dataframe_Human_cosine.csv', index_col=0)
        DM_MF_round = DM_MF.round(decimals=6)
        
        return DM_MF_round
    
    elif netlayout == 'funct-cel' and organism == 'human':
        
        DM_CC = pd.read_csv('input/DistanceMatrix_goCC_Dataframe_Human_cosine.csv', index_col=0)
        DM_CC_round = DM_CC.round(decimals=6)

        return DM_CC_round
    
    elif netlayout == 'funct-dis' and organism == 'human':

        DM_Disease = pd.read_csv('input/DistanceMatrix_Disease_Dataframe_Human_cosine.csv', index_col=0)
        DM_Disease_round= DM_Disease.round(decimals=6)

        return DM_Disease_round
    
    else: 
        print('Please type one of the following: "local", "global", "importance", "functional"')
























###### O L D #### F R O M #### H E R E #######



# ---------------------------------------------
# PLOTTING
# ---------------------------------------------

def draw_node_degree(G, scalef):
    #x = 20
    #ring_frac = np.sqrt((x-1.)/x)
    #ring_frac = (x-1.)/x

    l_size = {}
    for node in G.nodes():
        k = nx.degree(G, node)
        R = scalef * (1 + k**1.1) 

        l_size[node] = R
        
    return l_size


def draw_node_degree_3D(G, scalef):
    x = 3
    ring_frac = (x-1.)/x

    deg = dict(G.degree())
    
    d_size = {}
    for i in G.nodes():
        for k,v in deg.items():
            if i == k:
                R = scalef * (1+v**0.9)
                r = ring_frac * R
                d_size[i] = R
    
    return d_size 



def get_trace_nodes_2D(posG, info_list, color_list, size):

    key_list=list(posG.keys())
    trace = pgo.Scatter(x=[posG[key_list[i]][0] for i in range(len(key_list))],
                           y=[posG[key_list[i]][1] for i in range(len(key_list))],
                           mode = 'markers',
                           text = info_list,
                           hoverinfo = 'text',
                           #textposition='middle center',
                           marker = dict(
                color = color_list,
                size = size,
                symbol = 'circle',
                line = dict(width = 1.0,
                        color = color_list)
            ),
        )
    
    return trace


def get_trace_edges_2D(G, posG, color_list, opac = 0.2):
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = posG[edge[0]]
        x1, y1 = posG[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
 
    trace_edges = pgo.Scatter(
                        x = edge_x, 
                        y = edge_y, 
                        mode = 'lines', hoverinfo='none',
                        line = dict(width = 0.2, color = color_list),
                        opacity = opac
                )
    
    return trace_edges



def get_trace_nodes_3D(posG, info_list, color_list, size):

    key_list=list(posG.keys())
    trace = pgo.Scatter3d(x=[posG[key_list[i]][0] for i in range(len(key_list))],
                           y=[posG[key_list[i]][1] for i in range(len(key_list))],
                           z=[posG[key_list[i]][2] for i in range(len(key_list))],
                           mode = 'markers',
                           text = info_list,
                           hoverinfo = 'text',
                           #textposition='middle center',
                           marker = dict(
                color = color_list,
                size = size,
                symbol = 'circle',
                line = dict(width = 1.0,
                        color = color_list)
            ),
        )
    
    return trace


def get_trace_edges_3D(G, posG, color_list, opac = 0.2):
    edge_x = []
    edge_y = []
    edge_z = []
    for edge in G.edges():
        x0, y0, z0 = posG[edge[0]]
        x1, y1, z1 = posG[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        edge_z.append(z0)
        edge_z.append(z1)
        edge_z.append(None)

    trace_edges = pgo.Scatter3d(
                        x = edge_x, 
                        y = edge_y, 
                        z = edge_z,
                        mode = 'lines', hoverinfo='none',
                        line = dict(width = 0.1, color = color_list),
                        opacity = opac
                )
    
    return trace_edges


# ---------------------------------------------
# VISUAL PROPERTIES 
# ---------------------------------------------

def color_nodes_from_dict(G, d_to_be_coloured, color_method):

    #color_method = 'clos'
    #d_to_be_coloured = d_clos # dict sorted by dict.values (that way the biggest value matches darkest colour of palette)

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
    
    return colours