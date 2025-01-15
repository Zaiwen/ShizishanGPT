from flask import Flask, request,Response,jsonify
from flask.json import jsonify
from agent import Agent
from _get_res import getanswer
app = Flask(__name__)

agent = Agent()
app.config['JSON_AS_ASCII'] = False

@app.route('/invoke', methods=['GET'])  # 定义一个具体的路由路径
def get_invoke():
    string1 = request.args.get('question')  # 从查询参数中获取 'question'
    print(string1)
    return agent.query(string1) if string1 else "No question provided"

# @app.route('/CropGPT', methods=['GET']) 
# def get_res():
#     question = request.args.get('question')
#     _type,response = J.getAttribute(question)
    
#     json_response = jsonify({'type': _type, 'response': response})
#     return json_response

@app.route('/crop', methods=['GET'])  # 定义一个具体的路由路径
def get_answer():
    string1 = request.args.get('question')  # 从查询参数中获取 'question'
    senario = request.args.get('senario') # 获取场景信息
    print(string1)
    if string1:
        str = getanswer(string1,senario)
        return str
    else: return "No question provided"
    

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=5003)