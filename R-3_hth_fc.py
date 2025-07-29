# %%
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from plot_utils import divergent_green_orange

# %%
# load hth fc
individual_df = pd.read_csv('data/hth_individual_fc_Schaefer400_Fischl14.csv')
hth_fc = individual_df.iloc[:, 4:].values

# correlate the hth fc values
hth_fc_corr = np.corrcoef(hth_fc.T)
np.fill_diagonal(hth_fc_corr, np.nan)

# plot the hth fc correlation matrix
cmap = divergent_green_orange()
plt.figure(figsize=(10, 8), )
sns.heatmap(hth_fc_corr, cmap=cmap, center=0, square=True, vmin=0,
            cbar_kws={"label": 'HTH Similarity'})
plt.xlabel('Subjects')
plt.ylabel('Subjects')
plt.xticks([])
plt.yticks([])
plt.title('Interindividual Variability of Hypothalamic FC')

# %%
# How correlated are hth fc values with the mean across individuals?
hth_fc_mean = np.mean(hth_fc, axis=1)
hth_fc_mean_corr = np.array([spearmanr(hth_fc[:, i], hth_fc_mean)[0] for i in range(hth_fc.shape[1])])

# plot the hth fc mean correlation
plt.figure(figsize=(6,12),dpi=200)
sns.kdeplot(hth_fc_mean_corr)
plt.xlabel('Correlation with mean hypothalamic FC')
sns.despine()
plt.savefig('./figs/hth_fc_consistency.pdf')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                            AVERAGE BY NETWORK
###############################################################################
from plot_utils import divergent_green_orange
individual_df = pd.read_csv('data/hth_individual_fc_Schaefer400_Fischl14.csv')

# use the column 'network' to group the rows for each subject from sub-1 to sub-20
sub_cols = [f"sub-{i}" for i in np.arange(1, 21)]
plot_df = individual_df.groupby("network")[sub_cols].mean()

# define order of networks
network_order = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Cont', 'Default', 'Limbic' ] + \
                ['Amygdala', 'Caudate', 'Pallidum', 'Hippocampus', 'Accumbens', 'Putamen', 'Thalamus']

# reorder data and transpose to have genes as rows
plot_df = plot_df.loc[network_order]

fig, ax = plt.subplots(dpi=200)
sns.heatmap(plot_df, cmap=cmap, center=0, square=True, 
            cbar_kws={"shrink": .75, 'label': r'FC$_{HTH}$'}, vmin=0,
            ax=ax)
plt.ylabel(None)
