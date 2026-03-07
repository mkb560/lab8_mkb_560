import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys

filepath = sys.argv[1] 
stats = pd.read_csv(filepath)
if 'w2v' in filepath:
    model = 'Word2Vec'
else:
    model = 'Doc2Vec'

cluster_id = stats["cluster_id"].value_counts().index.tolist() # Cluster IDs
post_counts = stats["cluster_id"].value_counts().tolist() # Counts for each cluster

plt.figure(figsize=(10, 6))
sns.set_theme(style="white")

ax = sns.barplot(
    x=cluster_id, 
    y=post_counts, 
    palette="viridis",
    hue=cluster_id,
    legend=False
)

sns.despine(ax=ax, top=True, right=True)

ax.set_title(f"{model} Cluster Distribution", fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel("Cluster ID", fontsize=12, labelpad=10)
ax.set_ylabel("Number of Posts", fontsize=12, labelpad=10)

for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', 
                fontsize=11, color='black', 
                xytext=(0, 5), textcoords='offset points')
plt.tight_layout()
plt.show()