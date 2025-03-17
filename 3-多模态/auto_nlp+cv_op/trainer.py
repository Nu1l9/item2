from transformers import  DataCollatorForSeq2Seq, Trainer, TrainingArguments, \
    MT5Tokenizer, MT5ForConditionalGeneration
from datasets import Dataset
import transformers
import torch
import random
import logging
import shutil
logging.basicConfig(level=logging.INFO)

# 示例数据集
train_data = [
    {"text": "打开浏览器", "action_label": 1, "software_label": 0},
    {"text": "打开音乐播放器", "action_label": 1, "software_label": 1},
    {"text": "设置提醒", "action_label": 2, "software_label": 2},
    {"text": "打开文件夹", "action_label": 1, "software_label": 3},
    {"text": "打开微信", "action_label": 1, "software_label": 4},
    {"text": "设置音量", "action_label": 2, "software_label": 5},
    {"text": "打开计算器", "action_label": 1, "software_label": 6},
    {"text": "打开任务管理器", "action_label": 1, "software_label": 7},
    {"text": "关闭浏览器", "action_label": 0, "software_label": 0},
    {"text": "退出微信", "action_label": 0, "software_label": 4},
    ]

train_data += [
    {"text": "搜索百度", "action_label": "搜索信息", "software_label": "浏览器"},
    {"text": "在文件夹中查找文件", "action_label": "查找文件", "software_label": "文件管理器"},
    {"text": "搜索本地文档", "action_label": "查找文件", "software_label": "文件管理器"},
    {"text": "查找电子邮件", "action_label": "查找信息", "software_label": "邮件客户端"},
    {"text": "搜索网络新闻", "action_label": "搜索信息", "software_label": "浏览器"},
    {"text": "搜索视频教程", "action_label": "搜索信息", "software_label": "视频播放器"},
    {"text": "查找图片", "action_label": "查找文件", "software_label": "图片管理器"},
    {"text": "搜索歌曲", "action_label": "搜索信息", "software_label": "音乐播放器"},
    {"text": "在浏览器中查找网页", "action_label": "查找信息", "software_label": "浏览器"},
    {"text": "查找系统设置", "action_label": "查找信息", "software_label": "系统设置"}
]
train_data += [
    {"text": "创建新文件夹", "action_label": "文件管理", "software_label": "文件管理器"},
    {"text": "删除文件", "action_label": "文件管理", "software_label": "文件管理器"},
    {"text": "移动文件到另一个文件夹", "action_label": "文件管理", "software_label": "文件管理器"},
    {"text": "重命名文件", "action_label": "文件管理", "software_label": "文件管理器"},
    {"text": "复制文件", "action_label": "文件管理", "software_label": "文件管理器"},
    {"text": "粘贴文件", "action_label": "文件管理", "software_label": "文件管理器"},
    {"text": "查看文件属性", "action_label": "文件管理", "software_label": "文件管理器"},
    {"text": "压缩文件", "action_label": "文件管理", "software_label": "文件管理器"},
    {"text": "恢复已删除文件", "action_label": "文件管理", "software_label": "回收站"},
    {"text": "查看下载文件", "action_label": "文件管理", "software_label": "浏览器"}
]
train_data += [
    {"text": "输入文本", "action_label": "输入操作", "software_label": "文本编辑器"},
    {"text": "点击按钮", "action_label": "输入操作", "software_label": "应用程序"},
    {"text": "选择文件", "action_label": "输入操作", "software_label": "文件管理器"},
    {"text": "复制文本", "action_label": "输入操作", "software_label": "文本编辑器"},
    {"text": "粘贴文本", "action_label": "输入操作", "software_label": "文本编辑器"},
    {"text": "剪切文本", "action_label": "输入操作", "software_label": "文本编辑器"},
    {"text": "在搜索框中输入", "action_label": "输入操作", "software_label": "浏览器"},
    {"text": "键入密码", "action_label": "输入操作", "software_label": "账户管理"},
    {"text": "鼠标拖动文件", "action_label": "输入操作", "software_label": "文件管理器"},
    {"text": "在表格中输入数据", "action_label": "输入操作", "software_label": "办公软件"}
]

synonyms = {
    "打开": ["启动", "运行", "开启"],
    "浏览器": ["网页浏览器", "Chrome", "Edge"],
    "关闭": ["退出", "停止", "终止"]
}

def augment_text(text):
    words = text.split()
    for i in range(len(words)):
        if words[i] in synonyms:
            words[i] = random.choice(synonyms[words[i]])
    return "".join(words)

# 生成增强数据
augmented_data = []
for item in train_data:
    for _ in range(3):  # 每个样本生成3个变体
        new_text = augment_text(item['text'])
        augmented_data.append({
            "text": new_text,
            "action_label": item['action_label'],
            "software_label": item['software_label']
        })
train_data += augmented_data

# 加载T5 tokenizer
# ---- 模型初始化 ----
model_name = "google/mt5-base"
tokenizer = MT5Tokenizer.from_pretrained(model_name, legacy=False)
model = MT5ForConditionalGeneration.from_pretrained(model_name)

try:
    shutil.rmtree('./final_model')
except FileNotFoundError:
    pass


# ---- 修正预处理函数 ----
def preprocess_function(examples):
    # 输入格式
    inputs = ["指令：分析操作并识别软件。" + text for text in examples['text']]

    # 目标格式
    targets = [
        f"动作：{action}，软件：{software}"
        for action, software in zip(examples['action_label'], examples['software_label'])
    ]

    # Tokenize输入（关键修改：禁用return_tensors）
    model_inputs = tokenizer(
        inputs,
        max_length=64,
        padding=False,  # 禁用padding，由DataCollator处理
        truncation=True,
        return_tensors=None  # 必须返回列表格式
    )

    # Tokenize标签
    labels = tokenizer(
        targets,
        max_length=32,
        padding=False,
        truncation=True,
        return_tensors=None
    )

    # 替换padding token为-100
    model_inputs["labels"] = [
        [l if l != tokenizer.pad_token_id else -100 for l in label]
        for label in labels["input_ids"]
    ]

    return model_inputs


training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=30,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=2e-4,
    warmup_steps=200,
    weight_decay=0.01,
    logging_dir='./logs',
    eval_strategy="steps",  # 替换为 eval_strategy
    eval_steps=100,
    save_strategy="no",
    gradient_accumulation_steps=2,
    fp16=torch.cuda.is_available(),
    gradient_checkpointing=True,
    report_to="none",
    remove_unused_columns=False,
    max_grad_norm=1.0  # 梯度裁剪
)
# ---- 修正数据加载 ----
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    padding="longest",  # 动态padding
    return_tensors="pt"
)



trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# ---- 添加异常处理 ----
try:
    train_result = trainer.train()
except RuntimeError as e:
    print("\n===== 调试信息 =====")
    print("错误详情:", str(e))
    print("输入样本:", train_dataset[0])
    print("标签样本:", train_dataset[0]['labels'])
    raise

