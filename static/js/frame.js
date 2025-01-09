
class queryFormData{
  constructor(newskeycode2, _lucenesearchtype2, wbtreeid){
    this.newskeycode2 = newskeycode2
    this._lucenesearchtype = _lucenesearchtype2
    this.wbtreeid = wbtreeid
  }
}

const sendButton = document.querySelector(".send-button");
const bottomInput = document.querySelector(".right-area .bottom .text-input")
const topInput = document.querySelector(".right-area .top .tool-bar .search-area .search-input")
const topEnterEl = document.querySelector(".right-area .top .tool-bar .enter-icon")


// 全局监听Enter键点击
document.addEventListener('keydown', (event) => {
  if (event.key === 'Enter') {
      sendButton.click();
  }
});

// 聚焦时清除候选框文字——底部
bottomInput.addEventListener("focus", () => bottomInput.placeholder = "")
bottomInput.addEventListener("blur", () => bottomInput.placeholder = "请在此输入问题")

// 聚焦时清除候选框文字——顶部
topInput.addEventListener("focus", () => topInput.placeholder = "")
topInput.addEventListener("blur", () => topInput.placeholder = "搜索华农规章制度")

// 搜索规章制度
topEnterEl.onclick = () => {
  const base64Keyword= toBase64(topInput.value);
  const queryData = new queryFormData(base64Keyword, 2, 1001)
  const xhr = new XMLHttpRequest()
  xhr.open("POST", "http://127.0.0.1:5000/proxy");
  // xhr.open("POST", "http://218.199.69.86:5000/proxy");
  xhr.onload = function() {
    if(xhr.status >= 200 && xhr.status < 300 || xhr.status === 304){
          // 打开新标签页
          const newTab = window.open();
          // 将服务器返回的 HTML 写入新标签页
          newTab.document.write(xhr.responseText);
          newTab.document.close();  // 确保新标签页完成加载
      } else {
          console.error("查询请求失败: " + xhr.status);
    }
  };
  xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
  xhr.send(JSON.stringify(queryData));

  topInput.value = ""
}

// 将字符串转换为 Base64 编码
function toBase64(str) {
  const encoder = new TextEncoder();
  const data = encoder.encode(str);
  let binary = '';
  const len = data.byteLength;
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(data[i]);
  }
  return btoa(binary);
}

