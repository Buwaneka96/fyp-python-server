import json
from flask import Flask,redirect, url_for, Response, request
from Bert.predicter import predict_rules
from nlpComp import get_elements
from imageComparison import compareImages

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
   print(json.loads(request.data))
   return Response(json.dumps(predict_rules(json.loads(request.data))),  mimetype='application/json')

@app.route('/nlp/get_elements', methods=['POST'])
def elements():
   print(request.data)
   print(json.loads(request.data))
   return Response(json.dumps(get_elements(json.loads(request.data)[0])),  mimetype='application/json')

@app.route('/compare_images', methods=['POST'])
def compare():
   # print(json.loads(request.data))
   # return Response(json.dumps(predict_rules(json.loads(request.data))),  mimetype='application/json')   
   return Response(json.dumps(compareImages(json.loads(request.data)[0],json.loads(request.data)[1])),  mimetype='application/json')   

@app.route('/guest/<guest>')
def hello_guest(guest):
   return 'Hello %s as Guest' % guest

if __name__ == '__main__':
   app.run(debug = True)