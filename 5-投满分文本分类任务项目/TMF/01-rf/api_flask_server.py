# pip install flask
from flask import Flask,request,jsonify
from rf_predict_service import predict

# 1- 创建Application应用对象
app = Flask(__name__)

# API application process interface：URL地址，交换数据
# 2- 创建预测的后端API接口
"""
    POST和GET的区别：
        1- 安全性：POST相对GET要安全些
        2- 数据量：POST传输的数据量比GET要多
        3- 参数位置：GET在URL的后面，POST参数在表单中
"""
@app.route(rule="/predict_api",methods=["POST"])
def predict_api():
    # 1- 获得前端传递过来的参数
    # news_data的数据格式要求是 {"title":新闻标题内容}
    news_data = request.get_json()

    # 2- 调用预测方法
    result = predict(news_data)

    # 3- 返回结果给到前端
    return jsonify(result)

if __name__ == '__main__':
    # 启动后端程序
    """
       参数解释：
            host：程序运行的服务器IP地址
            port：程序绑定到服务器的什么端口号上。推荐设置范围是1024-65535之间 
    """
    app.run(host="127.0.0.1",port=8888,debug=True)