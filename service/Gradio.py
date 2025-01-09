
import gradio as gr

from ModelGPT.service.config.GradioConstants import TITLE, THEME, TEXTBOX_PLACEHOLDER, EXAMPLES, SUBMIT_BTN_TEXT, \
    RETRY_BTN_TEXT, UNDO_BTN_TEXT, CLEAR_BTN_TEXT, CONCURRENCY_LIMIT
from Service import Agent

# 创建 Gradio 界面

def create_gradio_interface():
    agent = Agent()
    interface = gr.ChatInterface(
        agent.query,
        type="messages",
        title=TITLE,
        theme=gr.themes.Default(**THEME),  # 使用默认主题
        textbox=gr.Textbox(placeholder=TEXTBOX_PLACEHOLDER, container=False, scale=7, elem_classes="textbox"),
        examples=EXAMPLES,  # 示例问题
        submit_btn=gr.Button(SUBMIT_BTN_TEXT, variant='primary', elem_classes="submit-btn"),  # 提交按钮的CSS类
        retry_btn=gr.Button(RETRY_BTN_TEXT),  # 重试按钮
        undo_btn=gr.Button(UNDO_BTN_TEXT),  # 撤回按钮
        clear_btn=gr.Button(CLEAR_BTN_TEXT),  # 清空记录按钮
        concurrency_limit=CONCURRENCY_LIMIT,  # 并发限制
    )
    return interface
