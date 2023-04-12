#flask application

from flask import Flask,request,render_template
from pipeline.predict_pipeline import PredictPipeline



application = Flask(__name__)
app = application
## route for a home page

@app.route('/')
def index():
    return render_template('home.html')


@app.route('/predict',methods=['GET','POST'])
def predict_datapoint():
    if request.method =='GET':
        return render_template('home.html')
    else:
        
        sentence = str(request.form.get('sentence'))
        print(sentence)

        Predict_pipeline=PredictPipeline()
        result = Predict_pipeline.predict(sentence)

        if result[0] == 3:
            result = 'English'
        elif result[0] == 4:                
            result = 'French'
        elif result[0] == 5:
            result = 'German'
        elif result[0] == 7:
            result = 'Hindi'
        elif result[0] == 8:
            result = 'Italian'
        elif result[0] == 12:
            result = 'Russian'
        elif result[0] == 13:
            result = 'Spanish'
        else:
            result = 'This Language is not supported'
        

        return render_template('home.html',result=result)


if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8000)
