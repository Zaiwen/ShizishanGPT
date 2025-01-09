'''
这个文件是一个Flask应用，提供了以下功能：
1. 启动一个Flask服务器，监听请求。
2. 根路径 ('/') 返回静态文件中的 index.html。
3. '/getMessage' 路径接受 POST 请求，处理用户输入的问题，并通过调用后端模型进行响应。
4. '/proxy' 路径代理请求，向外部站点发起 POST 请求并返回处理后的 HTML 内容。
'''

from bs4 import BeautifulSoup
import os
import service.Service as service
from flask_cors import CORS  # 导入CORS
from flask import Flask, request, jsonify, send_from_directory
import requests

# 设置要跑的卡
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
app = Flask(__name__, static_folder='static')
CORS(app)  # 启用CORS以允许跨域请求


# 根路径 '/' 直接返回静态文件中的 index.html
@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')

# 请求模型回答
@app.route('/getMessage', methods=['POST'])
def get_message():
    # 获取用户当前问题和对话历史记录（如无则默认为空列表）
    data = request.get_json()
    message = data.get('userMessage')
    history = data.get('history', [])

    # 调用模型的 service 函数进行响应
    response = service.answer(message, history)

    return jsonify({"content": response})

# 代理访问请求（用于访问华农官网内容）
@app.route('/proxy', methods=['POST'])
def proxy():
    data = request.get_json()
    print(data)
    target_url = "https://www.hzau.edu.cn/search.jsp?wbtreeid=1001&searchScope=0"
    response = requests.post(target_url, data=data) # 必须form-data，否则不解析
    print(response.text)
    # 解析 HTML，添加 <base> 标签,将所有相对路径指向目标站点的原始地址
    soup = BeautifulSoup(response.content, "html.parser")
    base_tag = soup.new_tag("base", href="https://www.hzau.edu.cn/")
    soup.head.insert(0, base_tag)
    # 清除 Transfer-Encoding 头
    headers = dict(response.headers)
    headers.pop("Transfer-Encoding", None)
    # return (response.content, response.status_code, {"Content-Type": "text/html"})
    return (str(soup), response.status_code, {"Content-Type": "text/html; charset=utf-8"})

# 运行 Flask 应用
if __name__ == '__main__':
    # Timer(1, open_browser).start()  # 延迟1秒打开，确保服务器先启动
    app.run(host="0.0.0.0",port=5001)
