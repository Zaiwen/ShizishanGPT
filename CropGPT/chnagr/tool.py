import requests
from bs4 import BeautifulSoup
# 中国农业科技信息网

def send_search_request(query, page):
    # 目标URL
    url = "https://cast.caas.cn/cms/web/search/index.jsp"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
    }
    # 要发送的POST请求数据
    data = {
        'matchType' : '1',
        'query': query,  # 搜索内容
        'page': page      # 当前页码
    }

    # 发送POST请求
    response = requests.post(url, data=data, headers=headers)

    # 检查请求是否成功
    if response.status_code == 200:
        response.encoding = response.apparent_encoding  # 设置正确的编码
        return response.text
    else:
        print("请求失败，状态码：", response.status_code)
        return None

def parse_search_results(html_content):
    # 使用BeautifulSoup解析HTML内容
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 找到所有搜索结果的容器
    results = soup.find_all('div', class_='search01')
    
    articles = []
    # 遍历所有搜索结果
    for result in results:
        # 获取标题
        title = result.find('h2').text.strip()
        # 获取发布日期
        date = result.find('span', recursive=False).text.strip()
        # 获取链接
        link = result.find('a')['href']
        
        # 获取文章内容
        content = get_article_content(link)
        
        # 保存文章信息
        articles.append({
            'title': title,
            'date': date,
            'link': link,
            'content': content
        })
        
        # 打印提取的内容
        print(f"标题: {title}")
        print(f"日期: {date}")
        print(f"链接: {link}")
        print(f"内容: {content[:100]}...")  # 打印内容的前100个字符作为预览
        print("-" * 80)  # 分隔线
    
    return articles

def get_article_content(link):
    # 发送GET请求获取文章页面内容
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
    }
    response = requests.get(link, headers=headers)
    if response.status_code == 200:
        response.encoding = response.apparent_encoding  # 设置正确的编码
        soup = BeautifulSoup(response.text, 'html.parser')

        res = ""
        # 提取文章内容
        content_div = soup.find('div', class_='zhengwen')
        if content_div:
            res += content_div.get_text(strip=True)
            print("res = ", res)
        else:
            print("未找到zhengwen标签，接下来找p=++++++++++++++++++")
            p_divs = soup.find_all('p')
            if p_divs:
                for p in p_divs:
                    res += p.get_text(strip=True) + "\n"
                print("res = ", res)
            else:
                print("未找到p标签============================")
        return res
    else:
        print("请求失败，状态码：", response.status_code)
    return ""

def get_total_pages(html_content):
    # 使用BeautifulSoup解析HTML内容
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 查找包含总页数的div标签
    pages_div = soup.find('div', class_='pages')
    print("标签 =============== ", pages_div)
    if pages_div:
        # 在div中找到包含页数的span标签
        pages_span = pages_div.find('span')
        if pages_span:
            # 提取总页数的数字部分，并转换为整数
            total_pages_text = pages_span.text.strip()
            return int(total_pages_text)
    # 如果没有找到总页数，返回1表示至少有1页
    return 1

def excute(query):
    page = 1
    total_pages = get_total_pages(send_search_request(query, page))
    print(f"总页数: {total_pages}")
    # 只爬取前2页的内容
    total_pages = total_pages if total_pages < 2 else 2

    all_articles = []
    while page <= total_pages:
        html_content = send_search_request(query, page)
        if html_content:
            articles = parse_search_results(html_content)
            all_articles.extend(articles)
        page += 1
    
    # 将所有文章信息保存到文件中
    with open('content.txt', 'w', encoding='utf-8') as f:
        for article in all_articles:
            f.write(f"标题: {article['title']}\n")
            f.write(f"日期: {article['date']}\n")
            f.write(f"链接: {article['link']}\n")
            f.write(f"内容: {article['content']}\n")
            f.write("-" * 80 + "\n")