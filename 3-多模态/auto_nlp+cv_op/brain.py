from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import re


from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import re

def predict_labels(text, model_path='./final_model'):
    # 加载模型
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)

    # 必须与训练时的输入格式完全一致！
    input_text = f"任务化转换: {text}"  # 关键修改点

    inputs = tokenizer(
        input_text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    # 生成时使用与训练一致的参数
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=128,
            num_beams=3,
            early_stopping=True,
            repetition_penalty=2.5  # 添加重复惩罚
        )

    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"模型原始输出: {decoded_output}")  # 调试输出

    # 增强版正则解析
    action_match = re.search(r'action[：:]\s*([^,;]+)', decoded_output, re.IGNORECASE)
    software_match = re.search(r'software[：:]\s*([^,;]+)', decoded_output, re.IGNORECASE)

    action_label = action_match.group(1).strip() if action_match else "未知"
    software_label = software_match.group(1).strip() if software_match else "未知"

    return action_label, software_label


# 示例测试
text = "打开浏览器"
action, software = predict_labels(text)
print(f"Action: {action}, Software: {software}")