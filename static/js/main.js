class userMessageData{
  constructor(userMessage){
    this.userMessage = userMessage
  }
}

const userInput = document.querySelector('.text-input');
const messagesContainer = document.querySelector('.chat-box');
// const url = "http://127.0.0.1:5001/getMessage"
const url = "http://218.199.69.86:5001/getMessage"
let enSend = true
let loadingElement = null; // 存储加载动画元素

sendButton.addEventListener('click', () => {
    const userMessage = userInput.value;
    console.log(userMessage)
    if (userMessage.trim() === '' ) {
      alert("输入不能为空");
      return
    }
    if (!enSend) {
      alert("发送过快")
      return
    }
    // 禁止用户发送内容
    enSend = false

    // 显示用户消息
    appendUserMessage(userMessage);

    // 清空输入框
    userInput.value = '';

    // 加载等待动画
    loadingElement = appendRobotLoading()

    // 发送ajax请求
    getModelResponse(new userMessageData(userMessage));
});


function appendUserMessage(message) {
    const userMessageElement = document.createElement('div');
    userMessageElement.className = 'user-chat-item chat-item';
    userMessageElement.innerHTML = `
        <div class="text-container">
            <div class="text-content">${message}</div>
        </div>
        <div class="user-profile">
            <i class="icon-user-profile"></i>
        </div>
    `;
    messagesContainer.appendChild(userMessageElement);
}

// 发送ajax请求
function getModelResponse(userMessageData) {
  const xhr = new XMLHttpRequest()
  xhr.responseType = "json";
  xhr.open("POST", url);
  xhr.setRequestHeader("Content-type", "application/json; charset=utf-8");

  xhr.onload = function() {
    if(xhr.status >= 200 && xhr.status < 300 || xhr.status === 304){
      // 移除加载动画，并显示机器人消息
      appendRobotMessage(xhr.response.content);
    } else {
      console.log("error! " + xhr.statusText)
      enSend = true; // 允许再次发送消息
    }
  }

  xhr.onerror = function() {
    console.log("网络请求失败，请检查网络连接: ", + xhr.statusText);
    enSend = true; // 允许再次发送消息
  };

  const paramJson = JSON.stringify(userMessageData)
  xhr.send(paramJson)
}

// 添加等待动画
function appendRobotLoading(){
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
  const robotMessageElement = document.createElement('div');
  robotMessageElement.className = 'robot-chat-item chat-item';
    // 1.展示空字符串
  robotMessageElement.innerHTML = `
        <div class="robot-profile">
            <i class="icon-robot-profile"></i>
        </div>
        <div class="text-container">
            <div class="text-content loading-animation">思考中...</div>
        </div>
    `;
  messagesContainer.appendChild(robotMessageElement);
  return robotMessageElement; // 返回加载动画元素
}


function appendRobotMessage(message) {
  // 确保加载动画元素存在
  if (!loadingElement) return;
  // 1.移除加载动画
  const textContentEl = loadingElement.querySelector(".text-container .text-content");
  textContentEl.classList.remove('loading-animation');
  textContentEl.innerText = "";

   // 开始输出时滚动到底部
   messagesContainer.scrollTop = messagesContainer.scrollHeight;

  // 2.流式写入
  let messageBuffer = ""
  let i = 0;
  const interval = 30; // 每30毫秒输出一个字符
  let charCounter = 0; // 用于计数每十个字符

  function executeNext() {
    if(message === null){
      textContentEl.innerText = "对不起，暂时无法回答"
      enSend = true
      return
    }
      if (i < message.length) {
        messageBuffer += message[i]; // 逐个添加字符到 messageBuffer
        textContentEl.innerText = messageBuffer;
        i++;
        charCounter++
        // 每输出20个字符时滚动到底部
        if (charCounter === 20) {
          messagesContainer.scrollTop = messagesContainer.scrollHeight;
          charCounter = 0; // 重置计数器
        }
        timeoutId = setTimeout(executeNext, interval); // 继续下一个字符
      }else{ // 输出结束
          enSend = true
          messagesContainer.scrollTop = messagesContainer.scrollHeight;
      }
  }
    executeNext(); // 开始逐步输出
}

