import pandas as pd
data = {'Name': ["John", "Doe", "Jane", "Bill"],
        'Location': ["New York", "Los Angeles", "Chicago", "Houston"],
        'Age': [20, 21,25,36]}

data_pandas = pd.DataFrame(data)
print(data_pandas)