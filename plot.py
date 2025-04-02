import matplotlib.pyplot as plt
import re

# Raw training log
log_text = """Epoch [1/100]  Train Loss: 0.7406  Val Loss: 0.6742
  [*] Model saved.
Training epoch
Validating
Epoch [2/100]  Train Loss: 0.6861  Val Loss: 0.6349
  [*] Model saved.
Training epoch
Validating
Epoch [3/100]  Train Loss: 0.6405  Val Loss: 0.5898
  [*] Model saved.
Training epoch
Validating
Epoch [4/100]  Train Loss: 0.6091  Val Loss: 0.5904
Training epoch
Validating
Epoch [5/100]  Train Loss: 0.5915  Val Loss: 0.5402
  [*] Model saved.
Training epoch
Validating
Epoch [6/100]  Train Loss: 0.5476  Val Loss: 0.5379
  [*] Model saved.
Training epoch
Validating
Epoch [7/100]  Train Loss: 0.5446  Val Loss: 0.4897
  [*] Model saved.
Training epoch
Validating
Epoch [8/100]  Train Loss: 0.5010  Val Loss: 0.5213
Training epoch
Validating
Epoch [9/100]  Train Loss: 0.4810  Val Loss: 0.4970
Training epoch
Validating
Epoch [10/100]  Train Loss: 0.5017  Val Loss: 0.4643
  [*] Model saved.
Training epoch
Validating
Epoch [11/100]  Train Loss: 0.4829  Val Loss: 0.4685
Training epoch
Validating
Epoch [12/100]  Train Loss: 0.4889  Val Loss: 0.4847
Training epoch
Validating
Epoch [13/100]  Train Loss: 0.4714  Val Loss: 0.4947
Training epoch
Validating
Epoch [14/100]  Train Loss: 0.4764  Val Loss: 0.4700
Training epoch
Validating
Epoch [15/100]  Train Loss: 0.4670  Val Loss: 0.4738
Training epoch
Validating
Epoch [16/100]  Train Loss: 0.4515  Val Loss: 0.4747
Training epoch
Validating
Epoch [17/100]  Train Loss: 0.4482  Val Loss: 0.4602
  [*] Model saved.
Training epoch
Validating
Epoch [18/100]  Train Loss: 0.4396  Val Loss: 0.4631
Training epoch
Validating
Epoch [19/100]  Train Loss: 0.4423  Val Loss: 0.4584
  [*] Model saved.
Training epoch
Validating
Epoch [20/100]  Train Loss: 0.4378  Val Loss: 0.4575
  [*] Model saved.
Training epoch
Validating
Epoch [21/100]  Train Loss: 0.4324  Val Loss: 0.4563
  [*] Model saved.
Training epoch
Validating
Epoch [22/100]  Train Loss: 0.4302  Val Loss: 0.4597
Training epoch
Validating
Epoch [23/100]  Train Loss: 0.4268  Val Loss: 0.4549
  [*] Model saved.
Training epoch
Validating
Epoch [24/100]  Train Loss: 0.4291  Val Loss: 0.4581
Training epoch
Validating
Epoch [25/100]  Train Loss: 0.4246  Val Loss: 0.4534
  [*] Model saved.
Training epoch
Validating
Epoch [26/100]  Train Loss: 0.4195  Val Loss: 0.4558
Training epoch
Validating
Epoch [27/100]  Train Loss: 0.4179  Val Loss: 0.4501
  [*] Model saved.
Training epoch
Validating
Epoch [28/100]  Train Loss: 0.4152  Val Loss: 0.4533
Training epoch
Validating
Epoch [29/100]  Train Loss: 0.4123  Val Loss: 0.4514
Training epoch
Validating
Epoch [30/100]  Train Loss: 0.4101  Val Loss: 0.4475
  [*] Model saved.
Training epoch
Validating
Epoch [31/100]  Train Loss: 0.4080  Val Loss: 0.4469
  [*] Model saved.
Training epoch
Validating
Epoch [32/100]  Train Loss: 0.4072  Val Loss: 0.4457
  [*] Model saved.
Training epoch
Validating
Epoch [33/100]  Train Loss: 0.4055  Val Loss: 0.4483
Training epoch
Validating
Epoch [34/100]  Train Loss: 0.4041  Val Loss: 0.4452
  [*] Model saved.
Training epoch
Validating
Epoch [35/100]  Train Loss: 0.4023  Val Loss: 0.4476
Training epoch
Validating
Epoch [36/100]  Train Loss: 0.4012  Val Loss: 0.4448
  [*] Model saved.
Training epoch
Validating
Epoch [37/100]  Train Loss: 0.3996  Val Loss: 0.4433
  [*] Model saved.
Training epoch
Validating
Epoch [38/100]  Train Loss: 0.3980  Val Loss: 0.4417
  [*] Model saved.
Training epoch
Validating
Epoch [39/100]  Train Loss: 0.3965  Val Loss: 0.4429
Training epoch
Validating
Epoch [40/100]  Train Loss: 0.3953  Val Loss: 0.4396
  [*] Model saved.
Training epoch
Validating
Epoch [41/100]  Train Loss: 0.3941  Val Loss: 0.4388
  [*] Model saved.
Training epoch
Validating
Epoch [42/100]  Train Loss: 0.3926  Val Loss: 0.4375
  [*] Model saved.
Training epoch
Validating
Epoch [43/100]  Train Loss: 0.3912  Val Loss: 0.4382
Training epoch
Validating
Epoch [44/100]  Train Loss: 0.3899  Val Loss: 0.4370
  [*] Model saved.
Training epoch
Validating
Epoch [45/100]  Train Loss: 0.3886  Val Loss: 0.4358
  [*] Model saved.
Training epoch
Validating
Epoch [46/100]  Train Loss: 0.3874  Val Loss: 0.4372
Training epoch
Validating
Epoch [47/100]  Train Loss: 0.3861  Val Loss: 0.4344
  [*] Model saved.
Training epoch
Validating
Epoch [48/100]  Train Loss: 0.3849  Val Loss: 0.4337
  [*] Model saved.
Training epoch
Validating
Epoch [49/100]  Train Loss: 0.3838  Val Loss: 0.4351
Training epoch
Validating
Epoch [50/100]  Train Loss: 0.3825  Val Loss: 0.4328
  [*] Model saved.
Training epoch
Validating
Epoch [51/100]  Train Loss: 0.3814  Val Loss: 0.4315
  [*] Model saved.
Training epoch
Validating
Epoch [52/100]  Train Loss: 0.3802  Val Loss: 0.4311
  [*] Model saved.
Training epoch
Validating
Epoch [53/100]  Train Loss: 0.3791  Val Loss: 0.4299
  [*] Model saved.
Training epoch
Validating
Epoch [54/100]  Train Loss: 0.3780  Val Loss: 0.4306
Training epoch
Validating
Epoch [55/100]  Train Loss: 0.3769  Val Loss: 0.4290
  [*] Model saved.
Training epoch
Validating
Epoch [56/100]  Train Loss: 0.3758  Val Loss: 0.4283
  [*] Model saved.
Training epoch
Validating
Epoch [57/100]  Train Loss: 0.3747  Val Loss: 0.4276
  [*] Model saved.
Training epoch
Validating
Epoch [58/100]  Train Loss: 0.3736  Val Loss: 0.4270
  [*] Model saved.
Training epoch
Validating
Epoch [59/100]  Train Loss: 0.3726  Val Loss: 0.4268
  [*] Model saved.
Training epoch
Validating
Epoch [60/100]  Train Loss: 0.3715  Val Loss: 0.4261
  [*] Model saved.

"""
log_text2 = """
Epoch [1/100]  Train Loss: 0.3748  Val Loss: 0.3372
  [*] Model saved.
Training epoch
Validating
Epoch [2/100]  Train Loss: 0.2914  Val Loss: 0.2983
  [*] Model saved.
Training epoch
Validating
Epoch [3/100]  Train Loss: 0.2634  Val Loss: 0.2738
  [*] Model saved.
Training epoch
Validating
Epoch [4/100]  Train Loss: 0.2411  Val Loss: 0.2622
  [*] Model saved.
Training epoch
Validating
Epoch [5/100]  Train Loss: 0.2339  Val Loss: 0.2460
  [*] Model saved.
Training epoch
Validating
Epoch [6/100]  Train Loss: 0.2239  Val Loss: 0.2449
  [*] Model saved.
Training epoch
Validating
Epoch [7/100]  Train Loss: 0.2170  Val Loss: 0.2424
  [*] Model saved.
Training epoch
Validating
Epoch [8/100]  Train Loss: 0.2073  Val Loss: 0.2315
  [*] Model saved.
Training epoch
Validating
Epoch [9/100]  Train Loss: 0.2041  Val Loss: 0.2315
Training epoch
Validating
Epoch [10/100]  Train Loss: 0.2045  Val Loss: 0.2238
  [*] Model saved.
Training epoch
Validating
Epoch [11/100]  Train Loss: 0.1975  Val Loss: 0.2271
Training epoch
Validating
Epoch [12/100]  Train Loss: 0.1903  Val Loss: 0.2208
  [*] Model saved.
Training epoch
Validating
Epoch [13/100]  Train Loss: 0.1901  Val Loss: 0.2191
  [*] Model saved.
Training epoch
Validating
Epoch [14/100]  Train Loss: 0.1864  Val Loss: 0.2158
  [*] Model saved.
Training epoch
Validating
Epoch [15/100]  Train Loss: 0.1835  Val Loss: 0.2152
  [*] Model saved.
Training epoch
Validating
Epoch [16/100]  Train Loss: 0.1818  Val Loss: 0.2176
Training epoch
Validating
Epoch [17/100]  Train Loss: 0.1795  Val Loss: 0.2122
  [*] Model saved.
Training epoch
Validating
Epoch [18/100]  Train Loss: 0.1765  Val Loss: 0.2089
  [*] Model saved.
Training epoch
Validating
Epoch [19/100]  Train Loss: 0.1727  Val Loss: 0.2177
Training epoch
Validating
Epoch [20/100]  Train Loss: 0.1723  Val Loss: 0.2042
  [*] Model saved.
Training epoch
Validating
Epoch [21/100]  Train Loss: 0.1719  Val Loss: 0.2044
Training epoch
Validating
Epoch [22/100]  Train Loss: 0.1682  Val Loss: 0.2074
Training epoch
Validating
Epoch [23/100]  Train Loss: 0.1646  Val Loss: 0.2045
Training epoch
Validating
Epoch [24/100]  Train Loss: 0.1636  Val Loss: 0.2111
Training epoch
Validating
Epoch [25/100]  Train Loss: 0.1653  Val Loss: 0.2001
  [*] Model saved.
Training epoch
Validating
Epoch [26/100]  Train Loss: 0.1615  Val Loss: 0.2039
Training epoch
Validating
Epoch [27/100]  Train Loss: 0.1584  Val Loss: 0.2008
Training epoch
Validating
Epoch [28/100]  Train Loss: 0.1580  Val Loss: 0.2012
Training epoch
Validating
Epoch [29/100]  Train Loss: 0.1559  Val Loss: 0.1982
  [*] Model saved.
Training epoch
Validating
Epoch [30/100]  Train Loss: 0.1532  Val Loss: 0.1949
  [*] Model saved.
Training epoch
Validating
Epoch [31/100]  Train Loss: 0.1520  Val Loss: 0.1971
Training epoch
Validating
Epoch [32/100]  Train Loss: 0.1497  Val Loss: 0.2034
Training epoch
Validating
Epoch [33/100]  Train Loss: 0.1519  Val Loss: 0.1957
Training epoch
Validating
Epoch [34/100]  Train Loss: 0.1469  Val Loss: 0.2007
Training epoch
Validating
Epoch [35/100]  Train Loss: 0.1466  Val Loss: 0.1960
Training epoch
Validating
Epoch [36/100]  Train Loss: 0.1461  Val Loss: 0.1971
Training epoch
Validating
Epoch [37/100]  Train Loss: 0.1446  Val Loss: 0.1995
Training epoch
Validating
Epoch [38/100]  Train Loss: 0.1457  Val Loss: 0.1991
Training epoch
Validating
Epoch [39/100]  Train Loss: 0.1424  Val Loss: 0.1984
Training epoch
Validating
Epoch [40/100]  Train Loss: 0.1401  Val Loss: 0.1974
Training epoch
Validating
Epoch [41/100]  Train Loss: 0.1387  Val Loss: 0.1964
Training epoch
Validating
Epoch [42/100]  Train Loss: 0.1389  Val Loss: 0.2011
Training epoch
Validating
Epoch [43/100]  Train Loss: 0.1411  Val Loss: 0.1995
Training epoch
Validating
Epoch [44/100]  Train Loss: 0.1357  Val Loss: 0.1945
  [*] Model saved.
Training epoch
Validating
Epoch [45/100]  Train Loss: 0.1335  Val Loss: 0.1965
Training epoch
Validating
Epoch [46/100]  Train Loss: 0.1324  Val Loss: 0.1949
Training epoch
Validating
Epoch [47/100]  Train Loss: 0.1335  Val Loss: 0.1947
Training epoch
Validating
Epoch [48/100]  Train Loss: 0.1297  Val Loss: 0.1987
Training epoch
Validating
Epoch [49/100]  Train Loss: 0.1329  Val Loss: 0.1964
Training epoch
Validating
Epoch [50/100]  Train Loss: 0.1298  Val Loss: 0.1926
  [*] Model saved.
Training epoch
Validating
Epoch [51/100]  Train Loss: 0.1273  Val Loss: 0.1960
Training epoch
Validating
Epoch [52/100]  Train Loss: 0.1280  Val Loss: 0.1969
Training epoch
Validating
Epoch [53/100]  Train Loss: 0.1268  Val Loss: 0.1983
Training epoch
Validating
Epoch [54/100]  Train Loss: 0.1284  Val Loss: 0.1935
Training epoch
Validating
Epoch [55/100]  Train Loss: 0.1240  Val Loss: 0.1949
Training epoch
Validating
Epoch [56/100]  Train Loss: 0.1246  Val Loss: 0.2034
Training epoch
Validating
Epoch [57/100]  Train Loss: 0.1238  Val Loss: 0.1961
Training epoch
Validating
Epoch [58/100]  Train Loss: 0.1204  Val Loss: 0.1964
Training epoch
Validating
Epoch [59/100]  Train Loss: 0.1200  Val Loss: 0.1945
Training epoch
Validating
Epoch [60/100]  Train Loss: 0.1215  Val Loss: 0.2017
Training epoch
Validating
Epoch [61/100]  Train Loss: 0.1187  Val Loss: 0.2009
Training epoch
Validating
Epoch [62/100]  Train Loss: 0.1205  Val Loss: 0.2010
Training epoch
Validating
Epoch [63/100]  Train Loss: 0.1177  Val Loss: 0.1984
"""

# Extract losses from log
train_losses = []
val_losses = []

# Get all epoch entries
lines = log_text.strip().split('\n')
for line in lines:
    match = re.search(r'Epoch \[(\d+)/\d+\]\s+Train Loss: ([\d.]+)\s+Val Loss: ([\d.]+)', line)
    if match:
        epoch = int(match.group(1))
        train_loss = float(match.group(2))
        val_loss = float(match.group(3))
        train_losses.append((epoch, train_loss))
        val_losses.append((epoch, val_loss))

# Unpack for plotting
epochs, train_vals = zip(*train_losses)
_, val_vals = zip(*val_losses)

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_vals, label='Train Loss')
plt.plot(epochs, val_vals, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
