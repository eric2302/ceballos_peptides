# %%
from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import index_structure
from neuromaps.stats import compare_images

# %%
data = loadmat('data/paquola/hcp_rDCM_sch400.mat')
atlas_info = pd.read_csv('data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_LUT.csv').iloc[54:-1]
sc = np.load('data/template_parc-Schaefer400_desc-SC_wei.npy')

eff_mat = data['results'][0][0][0]
np.fill_diagonal(eff_mat, 0)
regions = pd.DataFrame(index=data['results'][0][0][-1][0])

# check if eff_mat is symmetric
print(np.allclose(eff_mat, eff_mat.T))

# %%
sns.heatmap(eff_mat, cmap='RdBu_r')

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
genes = index_structure(genes, structure='CTX')
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

eff_mat_asymmetry = np.linalg.norm(eff_mat - eff_mat.T, 'fro').astype(float)
denominator = np.linalg.norm(eff_mat, 'fro').astype(float)
rel_eff_mat_asymmetry = eff_mat_asymmetry / denominator

# show kde plot but only after 0 since its non-negative
plt.figure(figsize=(8, 5), dpi=200)
sns.kdeplot(all_norms, color='grey', label='Ligand-Receptor Matrices', fill=True, alpha=0.1)
plt.axvline(x=rel_eff_mat_asymmetry, color='orange', label='Effective Connectivity Matrix')
plt.legend(bbox_to_anchor=(.55, .95), frameon=False)
plt.xlabel('Relative asymmetry (Frobenius Norm)')
plt.ylabel('Density')
plt.title(f'Asymmetry of Ligand-Receptor Matrices')


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                  COMPARE EFFECTIVE CONNECTIVITY IN VISUAL CORTEX
######################################################################################
visual_network = np.where(atlas_info['network'] == 'Vis')[0]
eff_vis = eff_mat[visual_network][:, visual_network]


sst_pair = pairs_available[(pairs_available['Peptide'] == 'SST') &
                           (pairs_available['Receptor'].isin(['SSTR1', 'SSTR2']))]
sst_mat = all_mat[sst_pair.index.values]
sst_mat = np.mean(sst_mat, axis=0)
sst_vis = sst_mat[visual_network][:, visual_network]

anatomical_connections = sc > 0
anatomical_connections = anatomical_connections[visual_network][:, visual_network]

plt.figure(figsize=(5, 5), dpi=200)
sns.heatmap(sst_vis * anatomical_connections.astype(int),
            cmap='inferno', cbar=False, xticklabels=False, yticklabels=False)
plt.title('Somatostatin interaction')
plt.xlabel('Region')
plt.ylabel('Region')

plt.figure(figsize=(5, 5), dpi=200)
sns.heatmap(eff_vis * anatomical_connections.astype(int), 
            cmap='inferno', cbar=False, xticklabels=False, yticklabels=False, vmin=0)
plt.title('Effective connectivity')
plt.xlabel('Region')
plt.ylabel('Region')


plt.figure(figsize=(5, 5), dpi=200)
sns.regplot(x=sst_vis[anatomical_connections].flatten(), 
            y=eff_vis[anatomical_connections].flatten(), 
            scatter_kws={'s':5}, color='grey', ci=None)
plt.xlabel('Somatostatin interaction')
plt.ylabel('Effective connectivity')


# sum across rows in both matrices
eff_input = np.sum(eff_vis[anatomical_connections], axis=0)
eff_output = np.sum(eff_vis[anatomical_connections], axis=1)
sst_receptivity = np.sum(sst_vis, axis=0)
sst_broadcasting = np.sum(sst_vis, axis=1)


plt.figure(figsize=(5, 5))
sns.regplot(x=sst_receptivity, y=eff_input, scatter_kws={'s':5}, color='grey')
plt.xlabel('SSTR input')
plt.ylabel('EC input')

plt.figure(figsize=(5, 5))
sns.regplot(x=sst_broadcasting, y=eff_output, scatter_kws={'s':5}, color='grey')
plt.xlabel('SST output')
plt.ylabel('EC output')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                  COMPARE AFFERENTS WITH SST RECEPTOR INPUTS
##########################################################################################
afferents = eff_mat.sum(axis=1)
receptor_inputs = sst_mat.sum(axis=1)

afferents = afferents[visual_network]
receptor_inputs = receptor_inputs[visual_network]

plt.figure(figsize=(5, 5))
sns.regplot(x=afferents, y=receptor_inputs, scatter_kws={'s':2}, color='grey')
plt.xlabel('Afferent input')
plt.ylabel('Receptor input')





# %%
# for each pair correlate the sum of eff_mat with the sum of the ligand-receptor matrix
all_mat_sum = np.sum(all_mat, axis=1)
eff_mat_sum = np.sum(eff_mat, axis=1)
nulls_idx = np.load('data/vasa_schaefer400_fsaverage_spin_indices.npy')[:, :1000]
# use nulls to resample the effective connectivity matrix
eff_nulls = eff_mat_sum[nulls_idx]
rs = []

for i in range(all_mat.shape[0]):
    pair_name = pairs_available['Receptor'][i] + '-' + pairs_available['Peptide'][i]
    r, p = compare_images(all_mat_sum[i], eff_mat_sum, nulls=eff_nulls, metric='spearmanr', return_nulls=False)
    if p < 0.05:
        rs.append(r)
        plt.figure(figsize=(5, 5))
        sns.regplot(x=all_mat_sum[i], y=eff_mat_sum, ci=None, color='grey', scatter_kws={'s': 0})
        sns.scatterplot(x=all_mat_sum[i], y=eff_mat_sum, s=5, hue=atlas_info['network'], palette='tab10', legend=False)
        plt.title(f"{pair_name}\nr = {r:.2f}")
        plt.xlabel('Ligand output')
        plt.ylabel('Effective connectivity output')
rs = np.array(rs)



# %%
# look at visual network only
visual_network = np.where(atlas_info['network'] == 'Vis')[0]

all_mat_sum = np.sum(all_mat, axis=1)[:, visual_network]
eff_mat_sum = np.sum(eff_mat, axis=1)[visual_network]

rs = []

for i in range(all_mat.shape[0]):
    pair_name = pairs_available['Receptor'][i] + '-' + pairs_available['Peptide'][i]
    r, p = compare_images(all_mat_sum[i], eff_mat_sum, nulls=eff_nulls[visual_network], 
                          metric='spearmanr', return_nulls=False)
    if p < 0.05:
        rs.append(r)
        plt.figure(figsize=(5, 5))
        sns.regplot(x=all_mat_sum[i], y=eff_mat_sum, ci=None, color='grey')
        plt.text(f'{pair_name}\nr = {r:.2f}\n' + )
        plt.xlabel('Receptor input')
        plt.ylabel('Effective connectivity input')
rs = np.array(rs)

# %% contrast with nulls!
# load nulls
visual_network = np.where(atlas_info['network'] == 'Vis')[0]
sst_pair = pairs_available[(pairs_available['Peptide'] == 'SST') &
                           (pairs_available['Receptor'] == 'SSTR1')]
sst_mat_vis = all_mat[sst_pair.index[0]][visual_network][:, visual_network]
eff_mat_vis = eff_mat[visual_network][:, visual_network]
np.fill_diagonal(eff_mat_vis, np.nan)
np.fill_diagonal(eff_mat_vis, np.nan)

all_mat_sum = np.sum(sst_mat_vis, axis=0)
eff_mat_sum = np.sum(eff_mat_vis, axis=0)

rs = []

for i in range(all_mat.shape[0]):
    pair_name = pairs_available['Receptor'][i] + '-' + pairs_available['Peptide'][i]
    plt.figure(figsize=(5, 5))
    sns.regplot(x=all_mat_sum[i], y=eff_mat_sum, ci=None, color='grey')
    plt.title(f"{pair_name}")


# %%
# find sst-sstr1 in pairs_available
sst_pair = pairs_available[(pairs_available['Peptide'] == 'SST') &
                           (pairs_available['Receptor'] == 'SSTR1')]
sst_mat = all_mat[sst_pair.index[0]][visual_network][:, visual_network]
eff_mat_conn = eff_mat[visual_network][:, visual_network]

np.fill_diagonal(sst_mat, np.nan)
np.fill_diagonal(eff_mat_conn, np.nan)

plt.figure(figsize=(5, 5))
sns.regplot(x=sst_mat.flatten(), y=eff_mat_conn.flatten(), color='grey')
# %%
# create a mask for the top-right and bottom-left quadrant of sst_mat


half = sst_mat.shape[0] // 2
mask = np.zeros_like(sst_mat, dtype=bool)
mask[:half, half:] = True  # Top-right quadrant
mask[half:, :half] = True  # Bottom-left quadrant




