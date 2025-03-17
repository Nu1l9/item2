import torch


# 1. 加载词汇表
def load_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            word, idx = line.strip().split('\t')
            vocab[word] = int(idx)
    return vocab


# 2. 加载数据
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.readlines()
    return [line.strip() for line in data]


# 3. 数据集类
class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, src_data, tgt_data, src_vocab, tgt_vocab, unk_token="<unk>", pad_token="<pad>"):
        self.src_data = src_data
        self.tgt_data = tgt_data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.unk_token = unk_token
        self.pad_token = pad_token

        # 确保UNK和PAD标记在词汇表中
        self.src_vocab[self.unk_token] = len(self.src_vocab)
        self.tgt_vocab[self.unk_token] = len(self.tgt_vocab)
        self.src_vocab[self.pad_token] = len(self.src_vocab)
        self.tgt_vocab[self.pad_token] = len(self.tgt_vocab)

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        src = self.src_data[idx]
        tgt = self.tgt_data[idx]

        # 将词语转换为对应的索引，若词不在词汇表中，使用UNK标记
        src_idx = [self.src_vocab.get(word, self.src_vocab[self.unk_token]) for word in src.split()]
        tgt_idx = [self.tgt_vocab.get(word, self.tgt_vocab[self.unk_token]) for word in tgt.split()]

        return torch.tensor(src_idx), torch.tensor(tgt_idx)


# 4. 设置模型（Transformer模型示例）
class TransformerModel(torch.nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, nhead, num_layers):
        super(TransformerModel, self).__init__()
        self.embedding = torch.nn.Embedding(src_vocab_size, d_model)
        self.transformer = torch.nn.Transformer(d_model, nhead, num_layers)
        self.fc_out = torch.nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt):
        src_emb = self.embedding(src)
        tgt_emb = self.embedding(tgt)
        output = self.transformer(src_emb, tgt_emb)
        return self.fc_out(output)


# 5. 主函数
def train_model(model, train_loader, optimizer, criterion, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for src, tgt in train_loader:
            optimizer.zero_grad()
            output = model(src, tgt[:-1])  # 排除目标的最后一个词
            loss = criterion(output.view(-1, output.size(-1)), tgt[1:].view(-1))  # 排除源的第一个词
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")


# 6. 加载词汇表
src_vocab = load_vocab('cn.txt.vocab.tsv')
tgt_vocab = load_vocab('en.txt.vocab.tsv')

# 7. 加载数据
src_data = load_data('cn.txt')
tgt_data = load_data('en.txt')

# 8. 创建数据集和数据加载器
dataset = TranslationDataset(src_data, tgt_data, src_vocab, tgt_vocab)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# 9. 初始化模型、优化器和损失函数
model = TransformerModel(len(src_vocab), len(tgt_vocab), d_model=256, nhead=8, num_layers=6)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss(ignore_index=src_vocab["<pad>"])  # 忽略PAD标记

# 10. 训练模型
train_model(model, train_loader, optimizer, criterion)
