# %%
from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from utils import index_structure
from neuromaps.stats import compare_images
from plot_utils import divergent_green_orange

# %%
data = loadmat('data/paquola/hcp_rDCM_sch400.mat')
atlas_info = pd.read_csv('data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_LUT.csv')
sc = np.load('data/template_parc-Schaefer400_TianS4_desc-SC.npy')

eff_mat = data['results'][0][0][0]
np.fill_diagonal(eff_mat, 0)
regions = pd.DataFrame(index=data['results'][0][0][-1][0])

# %%
# load ligand-receptor pairs
pairs_available = pd.read_csv('data/peptide_receptor_ligand_pairs.csv', index_col=0)

# load receptor and ligand from qc
receptor_genes = pd.read_csv('data/receptor_filtered.csv', index_col=0).index.to_list()
precursor_genes = pd.read_csv('data/precursor_qc.csv', index_col=0).index.to_list()
genes_of_interest = receptor_genes + precursor_genes

# keep only pairs that are in the gene library
pairs_available = pairs_available[pairs_available['Peptide'].isin(genes_of_interest)]
pairs_available = pairs_available[pairs_available['Receptor'].isin(genes_of_interest)]
pairs_available.sort_values('Receptor', inplace=True, ignore_index=True)
pairs_available = pairs_available[['Receptor', 'Peptide']].reset_index(drop=True)

# load genes of interest from all genes in gene library
all_genes = pd.read_csv('data/abagen_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv', index_col=0)

# in all_genes column names, select only ones that are in genes_of_interest
genes = all_genes[all_genes.columns.intersection(genes_of_interest)]
genes = index_structure(genes, structure='CTX-SBCTX')
# %%
all_mat = []
all_norms = []

for index, row in pairs_available.iterrows():
    ligand = genes[row['Peptide']].values[:, np.newaxis]
    receptor = genes[row['Receptor']].values[:, np.newaxis]
    
    # make sure ligand and receptor have no zero values
    # if so, add minimal value to avoid division by zero
    if np.any(ligand == 0):
        ligand[np.where(ligand == 0)] = 1e-10
    if np.any(receptor == 0):
        receptor[np.where(receptor == 0)] = 1e-10
    
    mat = ligand @ receptor.T
    
    # create an asymmetric matrix
    mat_asymmetric = mat - mat.T
    mat_asymmetric[np.diag_indices_from(mat_asymmetric)] = 0
    
    asymmetry = np.linalg.norm(mat_asymmetric, 'fro').astype(float)
    denominator = np.linalg.norm(mat, 'fro').astype(float)
    rel_asymmetry = asymmetry / denominator
    
    all_mat.append(mat)
    all_norms.append(rel_asymmetry)

all_mat = np.array(all_mat)
all_norms = np.array(all_norms)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                  RL ASYMMETRY IN NPY-NPY5R PAIR
######################################################################################
visual_network = np.where(atlas_info['network'] == 'Vis')[0]

npy_pair = pairs_available[(pairs_available['Peptide'] == 'NPY') &
                           (pairs_available['Receptor'] == 'NPY5R')]
npy_mat = all_mat[npy_pair.index[0]]

cmap = divergent_green_orange()
plt.figure(figsize=(5, 5), dpi=300)
sns.heatmap(npy_mat, cmap=cmap, cbar=False, xticklabels=False, yticklabels=False, square=True)
plt.xlabel('NPY5R')
plt.ylabel('NPY')
# plt.savefig('figs/npy-npy5r.jpeg', dpi=300)

# now plot only the upper triangle
mask = np.zeros_like(npy_mat, dtype=bool)
mask[np.tril_indices_from(mask)] = True
upper_npy_mat = npy_mat.copy()
upper_npy_mat[mask] = np.nan
plt.figure(figsize=(5, 5), dpi=200)
sns.heatmap(upper_npy_mat, cmap=cmap, cbar=False, xticklabels=False, yticklabels=False, square=True)
plt.xlabel('NPY5R')
plt.ylabel('NPY')
# plt.savefig('figs/npy-npy5r_upper.pdf')

# now the lower triangle
mask = np.zeros_like(npy_mat, dtype=bool)
mask[np.triu_indices_from(mask)] = True
lower_npy_mat = npy_mat.copy()
lower_npy_mat[mask] = np.nan
plt.figure(figsize=(5, 5), dpi=200)
sns.heatmap(lower_npy_mat, cmap=cmap, cbar=False, xticklabels=False, yticklabels=False, square=True)
plt.xlabel('NPY5R')
plt.ylabel('NPY')
# plt.savefig('figs/npy-npy5r_lower.pdf')

# compare upper and lower triangles
plt.figure(figsize=(5, 5), dpi=200)
sns.regplot(x=lower_npy_mat.T.flatten(), y=upper_npy_mat.flatten(), color='grey', 
            scatter_kws={'s': 0.1, 'alpha':0.4}, ci=None, line_kws={'color': 'orange', 'linewidth': 2}) 
plt.plot([0, np.max(npy_mat)], [0, np.max(npy_mat)], color='black', linestyle='--', linewidth=2)
plt.xlabel('Sender interaction')
plt.ylabel('Receiver interaction')
sns.despine()
# plt.savefig('figs/npy-npy5r_sender-receiver_interaction.pdf')

# subtract the lower triangle from the upper triangle
diff_npy_mat = lower_npy_mat.T - upper_npy_mat
plt.figure(figsize=(5, 5), dpi=300)
sns.heatmap(diff_npy_mat, cmap=cmap, xticklabels=False, yticklabels=False, square=True,
            cbar=True, cbar_kws={'label': 'More B <--> More A'}, center=0, vmin=-0.4, vmax=0.4)
# plt.savefig('figs/npy-npy5r_sender-receiver_diff.eps')


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                  RL ASYMMETRY COMAPRED TO EFFECTIVE CONNECTIVITY
######################################################################################

mask = np.zeros_like(eff_mat, dtype=bool)
mask[np.tril_indices_from(mask)] = True
upper_eff_mat = eff_mat.copy()
upper_eff_mat[mask] = np.nan

mask = np.zeros_like(eff_mat, dtype=bool)
mask[np.triu_indices_from(mask)] = True
lower_eff_mat = eff_mat.copy()
lower_eff_mat[mask] = np.nan

diff_eff_mat = upper_eff_mat - lower_eff_mat.T
vmax = np.nanmax(diff_eff_mat)
plt.figure(figsize=(5, 5), dpi=300)
sns.heatmap(diff_eff_mat, cmap=cmap, xticklabels=False, yticklabels=False, square=True,
            cbar=True, cbar_kws={'label': 'Connectivity difference'}, center=0, vmin=-vmax, vmax=vmax)
# plt.savefig('figs/effective-connectivity_diff.eps')

# compare the effective connectivity asymmetry with receptor-ligand matrices
eff_mat_asymmetry = np.linalg.norm(eff_mat - eff_mat.T, 'fro').astype(float)
denominator = np.linalg.norm(eff_mat, 'fro').astype(float)
rel_eff_mat_asymmetry = eff_mat_asymmetry / denominator

# show kde plot of all pairs
plt.figure(figsize=(8, 5), dpi=200)
sns.kdeplot(all_norms, color='grey', label='Ligand-Receptor Matrices', fill=True, alpha=0.1)
# add vertical line for effective connectivity matrix
plt.axvline(x=rel_eff_mat_asymmetry, color='orange', label='Effective Connectivity Matrix')
plt.legend(bbox_to_anchor=(1, .95), frameon=False)
plt.xlabel('Relative asymmetry (Frobenius Norm)')
plt.ylabel('Density')
sns.despine()
# plt.savefig('figs/relative_asymmetry.pdf')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                  REPLICATE RESULTS FROM SONG ET AL. 2020
######################################################################################
# Song et al. showed in their 2020 Science Advance paper that increased somatostatin signaling 
# leads to an overall circuit signaling through dishinhibition of parvalbumin interneurons

visual_network = np.where(atlas_info['network'] == 'Vis')[0]
eff_vis = eff_mat[visual_network][:, visual_network]

# look for all ligands that contain the word 'SST' and receptor that contain 'SSTR' in pairs_available
sst_pairs = pairs_available[pairs_available['Peptide'].str.contains('SST')]
sst_pairs = sst_pairs[sst_pairs['Receptor'].str.contains('SSTR')]
sst_mat = all_mat[sst_pairs.index.values]
sst_mat = np.mean(sst_mat, axis=0)
sst_vis = sst_mat[visual_network][:, visual_network]

anatomical_connections = sc > 0
anatomical_connections = anatomical_connections[visual_network][:, visual_network]

plt.figure(figsize=(5, 5), dpi=200)
sns.heatmap(sst_vis * anatomical_connections.astype(int),
            cmap=cmap, cbar=False, xticklabels=False, yticklabels=False,
            square=True, center=0, mask=~anatomical_connections)
plt.title('Somatostatin interaction')
plt.xlabel('Region')
plt.ylabel('Region')
plt.savefig('figs/somatostatin_interaction_visual_cortex.eps')

plt.figure(figsize=(5, 5), dpi=200)
sns.heatmap(eff_vis * anatomical_connections.astype(int), 
            cmap=cmap, cbar=False, xticklabels=False, yticklabels=False,
            square=True, center=0, mask=~anatomical_connections)
plt.title('Effective connectivity')
plt.xlabel('Region')
plt.ylabel('Region')
plt.savefig('figs/effective_connectivity_visual_cortex.eps')

# compare the sum of the somatostatin interaction (sst output) 
# with the sum of the effective connectivity (efferent output)
eff_output = np.sum(eff_vis * anatomical_connections, axis=1)
sst_broadcasting = np.sum(sst_vis * anatomical_connections, axis=1)
r, p = pearsonr(sst_broadcasting, eff_output)

plt.figure(figsize=(5, 5), dpi=300)
sns.regplot(x=sst_broadcasting, y=eff_output, color='grey',
            line_kws={'color': 'black', 'linewidth': 2}, ci=None)
plt.text(0.1, 0.98, f'r = {r:.2f}\nP < 0.001', fontsize=12, ha='left', va='top', transform=plt.gca().transAxes)
plt.xlabel('Somatostatin output')
plt.ylabel('Effective output')
sns.despine()
plt.savefig('figs/somatostatin_effective_output.pdf')
