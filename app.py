from flask import Flask, render_template,url_for,request,send_file,jsonify
from flask_cors import CORS,cross_origin
import pandas as pd
import pickle
import urllib3, requests, json
import math
import numpy as np
app = Flask(__name__)
cors=CORS(app)
app.config['CORS_HEADERS']='Content-Type'
@app.route('/')
@cross_origin()
def home():
 return render_template('home.html')
@app.route('/predict/test', methods = ['POST','GET'])
@cross_origin()
def predict_test():
    apikey ="3mu87DQi7ghAt9fjh4DpKNUfy3ygoDkPXv2jLLBpQeIM"
    url="https://iam.bluemix.net/oidc/token"
    headers = { "Content-Type" : "application/x-www-form-urlencoded" }
    data    = "apikey=" + apikey + "&grant_type=urn:ibm:params:oauth:grant-type:apikey"
    IBM_cloud_IAM_uid = "bx"
    IBM_cloud_IAM_pwd = "bx"
    response  = requests.post( url, headers=headers, data=data, auth=( IBM_cloud_IAM_uid, IBM_cloud_IAM_pwd ) )
    iam_token = response.json()["access_token"]
    ml_instance_id = "7ac7d89f-73a1-4fe9-83e5-bf1bbf7cb2b9"
    header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + iam_token, 'ML-Instance-ID': ml_instance_id}
    print(request.data)
    option=request.json['option']
    option_val=int(request.json['option-val'])
    category=int(request.json['category'])
    checkout_price=float(request.json['checkout-price'])
    if option=="meal-id":
        payload_scoring = {
        "fields": ["week",
		"checkout_price",
		"base_price",
		"emailer_for_promotion",
		"homepage_featured",
		"city_code",
		"region_code",
		"op_area",
		"discount_per",
		"center_id_encoded",
		"meal_id_encoded",
		"center_type_encoded",
		"category_encoded",
		"cuisine_encoded",
		"week_sin",
		"week_cos"],
        "values": [[146, checkout_price, 159.11, 0, 0, 647, 56, 2.0, 0.6284960090503425, 23, option_val, 2, category, 3, -0.9510714097000258, 0.3089711534289336]]}
    else:
        payload_scoring = {
        "fields": ["week",
		"checkout_price",
		"base_price",
		"emailer_for_promotion",
		"homepage_featured",
		"city_code",
		"region_code",
		"op_area",
		"discount_per",
		"center_id_encoded",
		"meal_id_encoded",
		"center_type_encoded",
		"category_encoded",
		"cuisine_encoded",
		"week_sin",
		"week_cos"],
        "values": [[146, checkout_price, 159.11, 0, 0, 647, 56, 2.0, 0.6284960090503425, option_val, 22, 2, category, 3, -0.9510714097000258, 0.3089711534289336]]}

    
    print(payload_scoring)
    response_scoring = requests.post("https://us-south.ml.cloud.ibm.com/v3/wml_instances/7ac7d89f-73a1-4fe9-83e5-bf1bbf7cb2b9/deployments/4e2f9e34-51a9-4362-8ba5-1767c7eb5aac/online", json=payload_scoring, headers=header)
    print("Scoring response")
    print(json.loads(response_scoring.text))
    jsonResult = json.loads(response_scoring.text)  
    prediction = jsonResult['values'][0][0]
    if prediction<4.5:
        prediction=int(prediction)
    else:
        prediction=int(prediction)+1
    print(prediction)
    return jsonify(prediction)
    
    
    
@app.route('/predict', methods = ['POST'])
@cross_origin()
def predict():

    loaded_model=pickle.load(open('finalized_model.pkl', 'rb'))
    test=pd.read_csv('data/full_test_rect.csv')
    columns_to_train = ['week','week_sin','week_cos','checkout_price','base_price','discount_per','emailer_for_promotion','homepage_featured','city_code', 
                    'region_code','center_type_encoded','op_area','category_encoded','cuisine_encoded','center_id_encoded','meal_id_encoded']

    categorical_columns = ['emailer_for_promotion','homepage_featured','city_code', 'region_code','center_type_encoded','category_encoded','cuisine_encoded',
                       'center_id_encoded','meal_id_encoded']

    numerical_columns = [col for col in columns_to_train if col not in categorical_columns]
    X=test[categorical_columns + numerical_columns]
    predictionT=loaded_model.predict(X)
    predictionT=np.expm1(predictionT)
    idd=[]
    for i in range(test.shape[0]):
        idd.append(test['id'][i])
    pred=pd.DataFrame({'id':idd,'num_orders':predictionT})
    pred.num_orders=pred.num_orders.astype('int')
    #print(predictionT)
    full_test=pd.merge(test,pred,how='left',left_on='id',right_on='id')
    meal=list(full_test.meal_id.unique())
    meal_groupt=full_test.groupby('meal_id')
    order_sortt=list(meal_groupt['num_orders'].agg(np.sum))
    meal_order=pd.DataFrame({'meal_id':meal,'num_orders':order_sortt})
    meal_order.to_csv('outputs/meal_order.csv',index=False)
    centert=list(full_test.center_id.unique())
    c_groupt=full_test.groupby('center_id')
    c_ordert=list(c_groupt['num_orders'].agg(np.sum))
    center_order=pd.DataFrame({'center_id':centert,'num_orders':c_ordert})
    center_order.to_csv('outputs/center_order.csv',index=False)
    cuisinet=list(full_test.cuisine.unique())
    cuisine_ordert=list(full_test.groupby('cuisine')['num_orders'].agg(np.sum))
    cuisine_order=pd.DataFrame({'cuisine':cuisinet,'num_orders':cuisine_ordert})
    cuisine_order.to_csv('outputs/cuisine_order.csv',index=False)
    catt=full_test.category.unique()
    cat_ordert=list(full_test.groupby('category')['num_orders'].agg(np.sum))
    category_order=pd.DataFrame({'category':catt,'num_orders':cat_ordert})
    category_order.to_csv('outputs/category_order.csv',index=False)
    if request.method == 'POST':
        return render_template('result.html', tables = [full_test.to_html(classes='result')])
@app.route('/meal/train/predictions/', methods=['GET'])
@cross_origin()
def api_meal_train_json():
    full_merge=pd.read_csv('data/full_merge_rect.csv')
    meal=list(full_merge.meal_id.unique())
    meal_groupt=full_merge.groupby('meal_id')
    order_sortt=list(meal_groupt['num_orders'].agg(np.sum))
    meal_order=pd.DataFrame({'meal_id':meal,'num_orders':order_sortt})
    meal=meal_order.to_json('outputs/train/meal_order.json')
    with open('outputs/train/meal_order.json') as jsonfile:
        meal = json.load(jsonfile)
    return jsonify(meal)
@app.route('/meal/predictions/', methods=['GET'])
@cross_origin()
def api_meal_json():
    meal=pd.read_csv('outputs/meal_order.csv').to_json('outputs/meal_order.json')
    with open('outputs/meal_order.json') as jsonfile:
        meal = json.load(jsonfile)
    return jsonify(meal)
@app.route('/meal/predictions/download/', methods=['GET'])
@cross_origin()
def api_meal():
    return send_file('outputs/meal_order.csv',mimetype='text/csv',attachment_filename='meal_order.csv',as_attachment=True)
@app.route('/center/train/predictions/', methods=['GET'])
@cross_origin()
def api_center_train_json():
    full_merge=pd.read_csv('data/full_merge_rect.csv')
    centert=list(full_merge.center_id.unique())
    c_groupt=full_merge.groupby('center_id')
    c_ordert=list(c_groupt['num_orders'].agg(np.sum))
    center_order=pd.DataFrame({'center_id':centert,'num_orders':c_ordert})
    center=center_order.to_json('outputs/train/center_order.json')
    with open('outputs/train/center_order.json') as jsonfile:
        center = json.load(jsonfile)
    return jsonify(center)
@app.route('/center/predictions/', methods=['GET'])
@cross_origin()
def api_center_json():
    center=pd.read_csv('outputs/center_order.csv').to_json('outputs/center_order.json')
    with open('outputs/center_order.json') as jsonfile:
        center = json.load(jsonfile)
    return jsonify(center)
@app.route('/center/predictions/download/', methods=['GET'])
@cross_origin()
def api_center():
    return send_file('outputs/center_order.csv',mimetype='text/csv',attachment_filename='center_order.csv',as_attachment=True)
@app.route('/cuisine/train/predictions/', methods=['GET'])
@cross_origin()
def api_cuisine_train_json():
    full_merge=pd.read_csv('data/full_merge_rect.csv')
    cuisinet=list(full_merge.cuisine.unique())
    cuisine_ordert=list(full_merge.groupby('cuisine')['num_orders'].agg(np.sum))
    cuisine_order=pd.DataFrame({'cuisine':cuisinet,'num_orders':cuisine_ordert})
    cuisine=cuisine_order.to_json('outputs/train/cuisine_order.json')
    with open('outputs/train/cuisine_order.json') as jsonfile:
        cuisine = json.load(jsonfile)
    return jsonify(cuisine)
@app.route('/cuisine/predictions/', methods=['GET'])
@cross_origin()
def api_cuisine_json():
    cuisine=pd.read_csv('outputs/cuisine_order.csv').to_json('outputs/cuisine_order.json')
    with open('outputs/cuisine_order.json') as jsonfile:
        cuisine = json.load(jsonfile)
    return jsonify(cuisine)
@app.route('/cuisine/predictions/download/', methods=['GET'])
@cross_origin()
def api_cuisine():
    return send_file('outputs/cuisine_order.csv',mimetype='text/csv',attachment_filename='cuisine_order.csv',as_attachment=True)
@app.route('/category/train/predictions/', methods=['GET'])
@cross_origin()
def api_category_train_json():
    full_merge=pd.read_csv('data/full_merge_rect.csv')
    catt=full_merge.category.unique()
    cat_ordert=list(full_merge.groupby('category')['num_orders'].agg(np.sum))
    category_order=pd.DataFrame({'category':catt,'num_orders':cat_ordert})
    category=category_order.to_json('outputs/train/category_order.json')
    with open('outputs/train/category_order.json') as jsonfile:
        category = json.load(jsonfile)
    return jsonify(category)
@app.route('/category/predictions/', methods=['GET'])
@cross_origin()
def api_category_json():
    category=pd.read_csv('outputs/category_order.csv').to_json('outputs/category_order.json')
    with open('outputs/category_order.json') as jsonfile:
        category = json.load(jsonfile)
    return jsonify(category)
@app.route('/category/predictions/download/', methods=['GET'])
@cross_origin()
def api_category():
    return send_file('outputs/category_order.csv',mimetype='text/csv',attachment_filename='category_order.csv',as_attachment=True)

if __name__ == '__main__':
 app.run(host='localhost',port='8080',debug=True)