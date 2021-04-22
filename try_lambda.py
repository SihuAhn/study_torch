import pandas as pd
import numpy as np

lst = ['1', '2', '3']

test_lst = list(map(lambda x : lst[x].zfill(5), range(len(lst))))
frame = pd.DataFrame(lst, columns=['Names'])
test_frame = frame['Names'].apply(lambda x: x.zfill(5))


print(test_lst, '\n', test_frame)



