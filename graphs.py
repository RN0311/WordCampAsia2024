import pandas as pd
import numpy as np

data = pd.read_csv("/content/wordpress_active_plugins_dataset.csv")
data.head()
data = data.drop(columns=["preview_link", "donate_link"])

import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("wordpress_active_plugins_dataset.csv")



plt.figure(figsize=(12, 6))
tag_counts = data['tag'].value_counts().head(10)
sns.barplot(x=tag_counts.values, y=tag_counts.index)
plt.title('Top 10 Most Popular WordPress Plugin Tags')
plt.xlabel('Number of Plugins')
plt.ylabel('Tag')
plt.tight_layout()
plt.savefig('popular_plugins.png')
plt.close()

plt.figure(figsize=(12, 6))
downloads_by_tag = data.groupby('tag')['downloaded'].sum().sort_values(ascending=False).head(10)
sns.barplot(x=downloads_by_tag.values, y=downloads_by_tag.index)
plt.title('Top 10 Most Downloaded WordPress Plugin Tags')
plt.xlabel('Number of Downloads')
plt.ylabel('Tag')

plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1e6)}M' if x < 1e9 else f'{x/1e9:.1f}B'))
plt.tight_layout()
plt.savefig('downloads_by_tag.png')
plt.close()

print("\nTop 5 tags by number of plugins:")
print(tag_counts.head().to_string())
print("\nTop 5 tags by total downloads:")
print(downloads_by_tag.head().to_string())


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("wordpress_active_plugins_dataset.csv")


colors = sns.color_palette("husl", 10)  

sns.set_style("whitegrid")

plt.figure(figsize=(12, 6))
tag_counts = data['tag'].value_counts().head(10)
ax = sns.barplot(x=tag_counts.values, y=tag_counts.index, palette=colors)


plt.title('Top 10 Most Popular WordPress Plugin Tags', fontsize=14, pad=20)
plt.xlabel('Number of Plugins', fontsize=12)
plt.ylabel('Tag', fontsize=12)

for i, v in enumerate(tag_counts.values):
    ax.text(v, i, f' {v}', va='center', fontsize=10)

plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))
downloads_by_tag = data.groupby('tag')['downloaded'].sum().sort_values(ascending=False).head(10)

ax = sns.barplot(x=downloads_by_tag.values, y=downloads_by_tag.index, palette=colors[::-1])  # Reverse palette for variety


plt.title('Top 10 Most Downloaded WordPress Plugin Tags', fontsize=14, pad=20)
plt.xlabel('Number of Downloads', fontsize=12)
plt.ylabel('Tag', fontsize=12)

def format_number(x, pos):
    if x >= 1e9:
        return f'{x/1e9:.1f}B'
    return f'{x/1e6:.1f}M'

ax.xaxis.set_major_formatter(plt.FuncFormatter(format_number))

for i, v in enumerate(downloads_by_tag.values):
    if v >= 1e9:
        label = f' {v/1e9:.1f}B'
    else:
        label = f' {v/1e6:.1f}M'
    ax.text(v, i, label, va='center', fontsize=10)

plt.tight_layout()
plt.show()
