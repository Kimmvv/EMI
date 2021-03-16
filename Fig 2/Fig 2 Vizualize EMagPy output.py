import pandas as pd
import os
import matplotlib.pyplot as plt

os.getcwd()


df = pd.read_csv('EMImatrix.csv', index_col=0, sep=',')

        ### 3 rows & 3 columns ###

titles = [['HCP1', 'HCP2.5', 'HCP4'], 
          ['VCP1', 'VCP2.5', 'VCP4'],
          ['PRP1', 'PRP2.5', 'PRP4']]

num = [[5,8,11],
       [14,17,20],
       [23,24,27]]


colour = ["b","y","r"]

fig, axes = plt.subplots(nrows = 3, ncols = 3,  sharex=True, sharey=True)

for i, row in enumerate(axes):
    for j, cell in enumerate(row):
        if num[i][j] == 5:
            legend = ["0.1m","0.3m","0.5m"]
        else:
            legend='_nolegend_'
        cell.hist([df[df.columns[num[i][j]]],df[df.columns[num[i][j]+1]],df[df.columns[num[i][j]+2]]], bins='auto', alpha=1, rwidth=0.85,
                  label=legend, color=colour, histtype='step')
        cell.set_title(titles[i][j])
        if i == len(axes) - 1:
            cell.set_xlabel("EC$_a$ [mS/m]".format(j + 1))
        if j == 0:
            cell.set_ylabel("Frequency".format(i + 1))

fig.legend(bbox_to_anchor=(1.12, 0.5),loc='center right', borderaxespad=0., title = "Height")
plt.tight_layout()
