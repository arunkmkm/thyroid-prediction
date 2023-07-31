import pickle
import pandas as pd
filename = 'thyroid_model.pkl'
model = open('thyroid_model.pkl','rb')
forest= pickle.load(model)



row_to_predict = [[41,0.015,96,2.5,22,1,1,0,0,0,0]]
df = pd.DataFrame(row_to_predict, columns = forest.feature_names_in_)
print(forest.predict(df))
