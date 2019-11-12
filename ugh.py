import numpy as np
import pandas as pd


aaa = np.array([1,3,3])
bbb = np.array([6,6,6])
df = pd.DataFrame(np.array([aaa,bbb]).T, columns=["A", "B"])

print(df)


