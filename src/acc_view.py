import matplotlib.pyplot as plt
import matplotlib.text as txt
import pandas as pd

csv_x = 'model_log_5class'


df = pd.read_csv('{}.csv'.format(csv_x))

plt.clf()
plt.figure(figsize = (12, 12))
plt.plot(df.Epoch, df['Validation Accuracy'])
plt.plot(df.Epoch, df.Accuracy)
plt.legend()
plt.savefig('{}_acc_plot.pdf'.format(csv_x))
plt.show()
