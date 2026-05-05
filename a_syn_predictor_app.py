# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 17:41:37 2020

@author: Lucas
"""


#%% Importing libraries

from pathlib import Path
import pandas as pd
import pickle
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from mordred import Calculator, descriptors
from multiprocessing import freeze_support
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
import base64


#%% PAGE CONFIG

#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='PaSynAI - Predictor of α-Synuclein Aggregation Inhibitors', page_icon="📏", layout='wide')

######
# Function to put a picture as header   
def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

image = Image.open('cropped-header.png')
st.image(image)

st.write("[![Website](https://img.shields.io/badge/website-LIDeB-blue)](https://lideb.biol.unlp.edu.ar)[![Twitter Follow](https://img.shields.io/twitter/follow/LIDeB_UNLP?style=social)](https://twitter.com/intent/follow?screen_name=LIDeB_UNLP)")
#st.subheader("📌" "About Us")

#st.markdown("""
#We are a drug discovery team with an interest in the development of publicly available open-source customizable
#cheminformatics tools to be used in computer-assisted drug discovery. We belong to the Laboratory of Bioactive
#Research and Development (LIDeB) of the National University of La Plata (UNLP), Argentina. 
#Our research group is focused on computer-guided drug repurposingand rational discovery of new drug candidates
#to treat epilepsy and neglected tropical diseases.            
#""")


# Introduction
#---------------------------------#

st.title(':computer: _PaSynAI - Predictor of α-Synuclein Aggregation Inhibitors_ ')
st.title('_Predictor of α-Synuclein Aggregation Inhibitors_ ')

st.write("""

**It is a free web-application for prediction of inhibitors of α-Synuclein Aggregation**

α-Syn is a key molecular target for the search of new treatments for Parkinson's Disease (PD).
The molecular events triggering the disease involve the abnormal aggregation of α-Syn in dopaminergic neurons.


Interfering with α-Syn aggregation is a promising disease-modifying approach for the treatment of PD.


PaSynAI is a Web App that ensembles 6 linear models to classify molecules as α-Syn aggregation inhibitors or non-inhibitors. 

The tool uses the following packages [RDKIT](https://www.rdkit.org/docs/index.html), [Mordred](https://github.com/mordred-descriptor/mordred), [sci-kit learn](https://scikit-learn.org/), [Plotly](https://plotly.com/)

**Workflow:**
""")


image = Image.open('workflow_PaSynAI.png')
st.image(image, caption='PaSynAI workflow')


#---------------------------------#
# Sidebar - Collects user input features into dataframe
st.sidebar.header('Upload your SMILES')
st.sidebar.markdown("""
[Example TXT input file](https://raw.githubusercontent.com/Capigol/a_syn_testing/main/example_file.txt)        
""")

uploaded_file_1 = st.sidebar.file_uploader("Upload a TXT file with one SMILES per line", type=["txt"])


#%% Standarization by MOLVS ####
####---------------------------------------------------------------------------####

smiles_column = 0
allowed_atoms = {"C","H","N","O","P","S","F","Cl","Br","I","B"}

normalizer = rdMolStandardize.Normalizer()
reionizer = rdMolStandardize.Reionizer()
uncharger = rdMolStandardize.Uncharger()
largest_frag = rdMolStandardize.LargestFragmentChooser()
tautomer_enumerator = rdMolStandardize.TautomerEnumerator()

def remove_isotopes(mol):
    """Elimina etiquetas isotópicas"""
    for atom in mol.GetAtoms():
        atom.SetIsotope(0)
    return mol

def is_valid_molecule(mol):
    if mol.GetNumAtoms() < 3:
        return False
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol not in allowed_atoms:
            return False
    return True

def standardize_smiles(smiles):
    if pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = remove_isotopes(mol)
        mol = normalizer.normalize(mol)
        mol = largest_frag.choose(mol)
        mol = reionizer.reionize(mol)
        mol = uncharger.uncharge(mol)
        mol = tautomer_enumerator.Canonicalize(mol)
        Chem.SanitizeMol(mol)
    except:
        return None
    if not is_valid_molecule(mol):
        return None
    return Chem.MolToSmiles(mol, canonical=True)


#%% Calculating molecular descriptors
### ----------------------- ###

def calcular_descriptores(data):
    
    data1x = pd.DataFrame()
    data["SMILES_STANDARDIZED"] = data[smiles_column].apply(standardize_smiles)
    df_quasi_final_estandarizado = data.copy()
    
    smiles_final = list(df_quasi_final_estandarizado["SMILES_STANDARDIZED"])
        
    # df_quasi_final_estandarizado["Final_SMILES"] = smiles_final
    
    calc = Calculator(descriptors, ignore_3D=True) 
    t = st.empty()
   
    smiles_ok = []
    for i,smiles in enumerate(smiles_final):
        if __name__ == "__main__":
                if smiles != None:
                    try:
                        mol = Chem.MolFromSmiles(smiles)
                        freeze_support()
                        descriptor1 = calc(mol)
                        resu = descriptor1.asdict()
                        solo_nombre = {'NAME' : f'SMILES_{i+1}'}
                        solo_nombre.update(resu)

                        solo_nombre = pd.DataFrame.from_dict(data=solo_nombre,orient="index")
                        data1x = pd.concat([data1x, solo_nombre],axis=1, ignore_index=True)
                        smiles_ok.append(smiles)
                        t.markdown("Calculating descriptors for molecule: " + str(i +1) +"/" + str(len(smiles_final)))
                    except:
                        
                        st.write(f'\rMolecule {smiles} has been removed (molecule not allowed by Mordred descriptor)')
                else:
                    pass

    data1x = data1x.T
    descriptores = data1x.set_index('NAME',inplace=False).copy()
    descriptores = descriptores.reindex(sorted(descriptores.columns), axis=1)   
    descriptores.replace([np.inf, -np.inf], np.nan, inplace=True)
    descriptores = descriptores.apply(pd.to_numeric, errors = 'coerce') 
    descriptores["Smiles_OK"] = smiles_ok
    # descriptors_total = formal_charge_calculation(descriptores)
    
    return descriptores, smiles_ok

#%% Determining Applicability Domain (AD)

def applicability_domain(prediction_set_descriptors, descriptors_model):
    
    descr_training = pd.read_csv("models/" + "desc_training_set_a_syn_1.csv")
    desc = descr_training[descriptors_model]
    t_transpuesto = desc.T
    multi = t_transpuesto.dot(desc)
    inversa = np.linalg.inv(multi)
    
    # Luego la base de testeo
    desc_sv = prediction_set_descriptors.copy()
    sv_transpuesto = desc_sv.T
    
    multi1 = desc_sv.dot(inversa)
    sv_transpuesto.reset_index(drop=True, inplace=True) 
    multi2 = multi1.dot(sv_transpuesto)
    diagonal = np.diag(multi2)
    
    # valor de corte para determinar si entra o no en el DA
    
    h2 = 2*(desc.shape[1]/desc.shape[0])  ## El h es 2 x Num de descriptores dividido el Num compuestos training. Mas estricto
    h3 = 3*(desc.shape[1]/desc.shape[0])  ##  Mas flexible
    
    diagonal_comparacion = list(diagonal)
    resultado_palanca2 =[]
    for valor in diagonal_comparacion:
        if valor < h2:
            resultado_palanca2.append(True)
        else:
            resultado_palanca2.append(False)
    resultado_palanca3 =[]
    for valor in diagonal_comparacion:
        if valor < h3:
            resultado_palanca3.append(True)
        else:
            resultado_palanca3.append(False)         
    return resultado_palanca2, resultado_palanca3


#%% Removing molecules with na in any descriptor

def all_correct_model(descriptors_total,loaded_desc, smiles_list):
    
    total_desc = []
    for descriptor_set in loaded_desc:
        for desc in descriptor_set:
            if not desc in total_desc:
                total_desc.append(desc)
            else:
                pass
            
    X_final = descriptors_total[total_desc]
    X_final["SMILES_OK"] = smiles_list
    rows_with_na = X_final[X_final.isna().any(axis=1)]         # Find rows with NaN values
    for molecule in rows_with_na["SMILES_OK"]:
        st.write(f'\rMolecule {molecule} has been removed (NA value  in any of the necessary descriptors)')
    X_final1 = X_final.dropna(axis=0,how="any",inplace=False)
    
    smiles_final = X_final1["SMILES_OK"]
    return X_final1, smiles_final

 # Function to assign colors based on confidence values
def get_color(confidence):
    """
    Assigns a color based on the confidence value.

    Args:
        confidence (float): The confidence value.

    Returns:
        str: The color in hexadecimal format (e.g., '#RRGGBB').
    """
    # Define your color logic here based on confidence
    if confidence == "HIGH" or confidence == "Substrate":
        return 'lightgreen'
    elif confidence == "MEDIUM":
        return 'yellow'
    else:
        return 'red'


#%% Predictions        

def predictions(loaded_model, loaded_desc, X_final1):
    scores = []
    palancas2 = []
    palancas3 = []

    i = 0
    
    for estimator in loaded_model:
        descriptors_model = loaded_desc[i]
        
        X = X_final1[descriptors_model]
        predictions = estimator.predict(X)
    
        scores.append(predictions)
        resultado_palanca2, resultado_palanca3  = applicability_domain(X, descriptors_model)
        palancas2.append(resultado_palanca2)
        palancas3.append(resultado_palanca3)
        i = i + 1 
    
    dataframe_scores = pd.DataFrame(scores).T
    dataframe_scores.index = smiles_final
    
    palancas_final2 = pd.DataFrame(palancas2).T
    palancas_final2.index = smiles_final
    palancas_final2['Confidence'] = (palancas_final2.sum(axis=1) / len(palancas_final2.columns)) * 100
    
    palancas_final3 = pd.DataFrame(palancas3).T
    palancas_final3.index = smiles_final
    palancas_final3['Confidence3'] = (palancas_final3.sum(axis=1) / len(palancas_final3.columns)) * 100

    score_ensemble = dataframe_scores.min(axis=1)
    classification = score_ensemble >= 0.44
    classification = classification.replace({True: 'Substrate', False: 'Non Substrate'})
    
    final_file = pd.concat([classification,palancas_final2['Confidence'], palancas_final3['Confidence3']], axis=1)
    
    final_file.rename(columns={0: "Prediction"},inplace=True)

    final_file["Confidence"] = final_file["Confidence"].astype(object)

    final_file.loc[final_file["Confidence"] >= 50, 'Confidence'] = 'HIGH'
    final_file.loc[(final_file["Confidence3"] >= 50) & (final_file["Confidence"] != "HIGH"), 'Confidence'] = 'MEDIUM'
    final_file.loc[final_file["Confidence3"] < 50, 'Confidence'] = 'LOW'

    final_file.loc[final_file["Confidence3"] < 50, 'Prediction'] = 'No conclusive'
    final_file.drop(columns=['Confidence3'],inplace=True)
            
    df_no_duplicates = final_file[~final_file.index.duplicated(keep='first')]
    styled_df = df_no_duplicates.style.apply(lambda row: [f"background-color: {get_color(row['Confidence'])}" for _ in row],subset=["Confidence"], axis=1)
    
    return final_file, styled_df



#%% Create plot:




def final_plot(final_file):
    non_conclusives = len(final_file[final_file['Confidence'] == "LOW"]) 
    substrates_hc = len(final_file[(final_file['Confidence'] == "HIGH") & (final_file['Prediction'] == 'Substrate')])
    substrates_mc = len(final_file[(final_file['Confidence'] == "MEDIUM") & (final_file['Prediction'] == 'Substrate')])

    # Count values in 'DA' column higher than 50 and 'class' is 'no'
    non_substrates_hc = len(final_file[(final_file['Confidence'] == "HIGH") & (final_file['Prediction'] == 'Non Substrate')])
    non_substrates_mc = len(final_file[(final_file['Confidence'] == "MEDIUM") & (final_file['Prediction'] == 'Non Substrate')])
    keys = ["Substrate - High confidence", "Substrate - Medium confidence", "Non Substrate - High confidence", "Non Substrate - Medium confidence", "Non conclusive"]
    fig = go.Figure(go.Pie(labels=keys, values=[substrates_hc, substrates_mc, non_substrates_hc, non_substrates_mc, non_conclusives]))
        
    fig.update_layout(title_text=None)
    
    return fig


#%%
def filedownload1(df):
    csv = df.to_csv(index=True,header=True)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="PaSynAI_predictions.csv">Download CSV File with results</a>'
    return href

#%% CORRIDA

loaded_model = pickle.load(open("models/" + "a_syn_linear_regression.pickle", 'rb'))
loaded_desc = pickle.load(open("models/" + "a_syn_model_descriptors.pickle", 'rb'))


if uploaded_file_1 is not None:
    run = st.button("RUN")
    if run == True:
        data = pd.read_csv(uploaded_file_1,sep="\t",header=None)       
        descriptors_total, smiles_list = calcular_descriptores(data)
        X_final1, smiles_final = all_correct_model(descriptors_total,loaded_desc, smiles_list)
        final_file, styled_df = predictions(loaded_model, loaded_desc, X_final1)
        figure  = final_plot(final_file)  
        col1, col2 = st.columns(2)

        with col1:
            st.header("Predictions")
            st.write(styled_df)
        with col2:
            st.header("Resume")
            st.plotly_chart(figure,use_container_width=True)
        st.markdown(":point_down: **Here you can download the results**", unsafe_allow_html=True)
        st.markdown(filedownload1(final_file), unsafe_allow_html=True)
       

# Example file
else:
    st.info('👈🏼👈🏼👈🏼      Awaiting for TXT file to be uploaded.')
    if st.button('Press to use Example Dataset'):
        data = pd.read_csv("example_file.txt",sep="\t",header=None)
        descriptors_total, smiles_list = calcular_descriptores(data)
        X_final1, smiles_final = all_correct_model(descriptors_total,loaded_desc, smiles_list)
        final_file, styled_df = predictions(loaded_model, loaded_desc, X_final1)
        figure  = final_plot(final_file)  
        col1, col2 = st.columns(2)
        with col1:
            st.header("Predictions")
            st.write(styled_df)
        with col2:
            st.header("Resume")
            st.plotly_chart(figure,use_container_width=True)
  
        st.markdown(":point_down: **Here you can download the results**", unsafe_allow_html=True)
        st.markdown(filedownload1(final_file), unsafe_allow_html=True)

#Footer edit

footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}
a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}
.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Made in  🐍 and <img style='display: ; 
' href="https://streamlit.io" src="https://i.imgur.com/iIOA6kU.png" target="_blank"></img> Developed with ❤️ by <a style='display: ;
 text-align: center' href="https://twitter.com/capigol" target="_blank">Lucas Alberca</a> for <a style='display: ; 
 text-align: center;' href="https://lideb.biol.unlp.edu.ar/" target="_blank">LIDeB</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)

