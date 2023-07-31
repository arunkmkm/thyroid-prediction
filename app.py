from flask import Flask, render_template, request
import pickle
import pandas as pd
filename = 'thyroid_model.pkl'
model = open('thyroid_model.pkl','rb')
forest= pickle.load(model)
dic={"YES":1,"NO":0,"MALE":1,"FEMALE":0}
lst=['Compensated','Negative','Primary','Secondary']
app = Flask(__name__,static_folder='templates/assets/')
@app.route('/', methods=['GET', 'POST'])
def open():
    return render_template('index_1.html')
@app.route('/test', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        data = []
        for i in range(1, 12):
            input_name = f'input_{i}'
            input_value = request.form.get(input_name)
            print(input_value)
            try:
                data.append(float(input_value))
            except:
                data.append(dic[input_value])
                print(data)
        # Do something with the data, e.g. save it to a database
        print(data)
        
            
        row_to_predict = [data]
        df = pd.DataFrame(row_to_predict, columns = forest.feature_names_in_)
        return render_template('result.html',diagnosis=lst[int(forest.predict(df))])
    return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True,host='192.168.92.130',port=5000)
