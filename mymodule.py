import time,sys,os
import torch,math
from tqdm import tqdm
from torch import nn, optim
import torch.nn.functional as F
import torchvision,collections
import zipfile,random
import torchtext.vocab as Vocab
import torch.utils.data as Data
sys.path.append("..")

# 基本卷积操作(互相关)
def corr2d(X, K): 
    h, w = K.shape
    Y = torch.zeros((X.shape[0]-h+1, X.shape[1]-w+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j] = (X[i:i+h, j:j+w] * K).sum()
    return Y

class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))
    def forward(self, x):
        return corr2d(x, self.weight) + self.bias

# 全局平均池化层
class GlobalAvgPool2d(nn.Module):
 # 全局平均池化层可通过将池化窗⼝形状设置成输⼊的⾼和宽实现
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])

# 压平:一个维度变为1
class FlattenLayer(nn.Module):
    def __init__(self):
       super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
       return x.view(x.shape[0], -1)

def load_data_fashion_mnist(batch_size, resize=None, root='~/Datasets/FashionMNIST'):
    """Download the fashion mnist dataset and then load into memory."""
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())
    
    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
    if sys.platform.startswith('win'):
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    else:
        num_workers = 4
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter

def evaluate_accuracy(data_iter, net, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train() # 改回训练模式
            else: # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item() 
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item() 
            n += y.shape[0]
    return acc_sum / n

def train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

def load_data_jay_lyrics():
    """加载周杰伦歌词数据集"""
    with zipfile.ZipFile('./data/lyrics/jaychou_lyrics.txt.zip') as zin:
        with zin.open('jaychou_lyrics.txt') as f:
            corpus_chars = f.read().decode('utf-8')
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    corpus_chars = corpus_chars[0:10000]
    # print(type(corpus_chars))
    # idx_to_char是每一个字的列表
    idx_to_char = list(set(corpus_chars)) # 转为set是为了去重
    # char_to_idx是每一个字加上去重后的位置序号的字典
    # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    # print(char_to_idx)
    # vocab_size是总字数
    vocab_size = len(char_to_idx) 
    # corpus_indices是一个列表,列表里的每个元素是与数据集中每个字符相对应的id(去重后的位置序号)
    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    return corpus_indices, char_to_idx, idx_to_char, vocab_size

# 索引为0的ONE-HOT向量
def one_hot(x, n_class, dtype=torch.float32):
    # X shape: (batch), output shape: (batch, n_class)
    x = x.long()
    res = torch.zeros(x.shape[0], n_class, dtype=dtype,device=x.device)
    # scatter_(dim,index,src)将src中数据根据index中的索引按照dim的方向填进input中
    res.scatter_(1, x.view(-1, 1), 1)
    return res
# 索引为2的ONE-HOT向量
def to_onehot(X, n_class): 
    # X shape: (batch, seq_len), output: seq_len elements of (batch, n_class)
    return [one_hot(X[:, i], n_class) for i in range(X.shape[1])]
'''
X = torch.arange(10).view(2, 5) #5个x1x2,每一个有1027种可能
inputs = to_onehot(X, 1027)
print(len(inputs), inputs[0].shape) # 5 torch.Size([2, 1027])
'''

# 每次从数据⾥随机采样⼀个⼩批量.其中批量⼤小batch_size指每个⼩批量的样本数,num_steps为每个样本所包含的时间步数。
def data_iter_random(corpus_indices, batch_size, num_steps, device=None):
    # 减1是因为输出的索引x是相应输入的索引y加1
    num_examples = (len(corpus_indices) - 1) // num_steps # 整除
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices) # 随机

    # 返回从pos开始的长为num_steps的序列
    def _data(pos):
        return corpus_indices[pos: pos + num_steps]
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for i in range(epoch_size):
        # 每次读取batch_size个随机样本
        i = i * batch_size
        batch_indices = example_indices[i: i + batch_size]
        X = [_data(j * num_steps) for j in batch_indices]
        Y = [_data(j * num_steps + 1) for j in batch_indices] # Y比X多一个时间步 
        yield torch.tensor(X, dtype=torch.float32, device=device), torch.tensor(Y, dtype=torch.float32, device=device)

def data_iter_consecutive(corpus_indices, batch_size, num_steps, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    corpus_indices = torch.tensor(corpus_indices, dtype=torch.float32, device=device)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    indices = corpus_indices[0: batch_size*batch_len].view(batch_size, batch_len)
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y

# 随机梯度下降法
def sgd(params, lr, batch_size):
    # 为了和原书保持一致，这里除以了batch_size，但是应该是不用除的，因为一般用PyTorch计算loss时就默认已经沿batch维求了平均了。
    for param in params:
        param.data -= lr * param.grad / batch_size # 注意这里更改param时用的param.data

# 梯度裁剪
def grad_clipping(params, theta, device):
    norm = torch.tensor([0.0], device=device)
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta / norm)

def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state,num_hiddens, vocab_size, device, idx_to_char,char_to_idx):
    state = init_rnn_state(1, num_hiddens, device)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        # 将上⼀时间步的输出作为当前时间步的输⼊
        X = mymodule.to_onehot(torch.tensor([[output[-1]]], device=device),vocab_size)
        # 计算输出和更新隐藏状态
        (Y, state) = rnn(X, state, params)
        # 下⼀个时间步的输⼊是prefix⾥的字符或者当前的最佳预测字符
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y[0].argmax(dim=1).item()))
    return ''.join([idx_to_char[i] for i in output])

def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                          vocab_size, device, corpus_indices, idx_to_char,
                          char_to_idx, is_random_iter, num_epochs, num_steps,
                          lr, clipping_theta, batch_size, pred_period,
                          pred_len, prefixes):
    
    data_iter_fn = mymodule.data_iter_random # 随机采样

    params = get_params()
    loss = nn.CrossEntropyLoss() # 交叉熵损失函数

    for epoch in range(num_epochs):
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, device) #采样
        for X, Y in data_iter:
            # 使用随机采样，在每个小批量更新前初始化隐藏状态
            state = init_rnn_state(batch_size, num_hiddens, device)
            inputs = mymodule.to_onehot(X, vocab_size)
            # outputs有num_steps个形状为(batch_size, vocab_size)的矩阵
            (outputs, state) = rnn(inputs, state, params)
            # 拼接之后形状为(num_steps * batch_size, vocab_size)
            outputs = torch.cat(outputs, dim=0)
            # Y的形状是(batch_size, num_steps)，转置后再变成长度为
            # batch * num_steps 的向量, 这样跟输出的行一一对应
            y = torch.transpose(Y, 0, 1).contiguous().view(-1)
            # 使用交叉熵损失计算平均分类误差
            l = loss(outputs, y.long()) #Y是采样出来的下一步的真实结果
            
            # 梯度清0:调用backward()函数之前都要将梯度清零，因为如果梯度不清零，pytorch中会将上次计算的梯度和本次计算的梯度累加。
            if params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            l.backward() #向后传播,自动求导
            mymodule.grad_clipping(params, clipping_theta, device)  # 裁剪梯度
            mymodule.sgd(params, lr, 1)  # 梯度下降,因为误差已经取过均值,梯度不用再做平均
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        # 预测
        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn(prefix, pred_len, rnn, params, init_rnn_state,
                    num_hiddens, vocab_size, device, idx_to_char, char_to_idx))
                    
def train(train_iter, test_iter, net, loss, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y) 
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

# 读取训练数据集和测试数据集。每个样本是一条评论及其对应的标签：1正0负
def read_imdb(folder='train', data_root="./data/"): 
    data = []
    for label in ['pos', 'neg']:
        folder_name = os.path.join(data_root, folder, label)
        # 列出当前文件夹下的文件,tqdm是进度条库
        for file in tqdm(os.listdir(folder_name)): # 一个文件一条句子
            with open(os.path.join(folder_name, file), 'rb') as f:
                sentence = f.read().decode('utf-8').replace('\n', '').lower()
                data.append([sentence, 1 if label == 'pos' else 0])
    random.shuffle(data) # 随机打乱 
    return data
# 基于空格进行分词
# 本函数已保存在d2lzh_pytorch包中方便以后使用
def get_tokenized_imdb(data):
    """
    data: list of [string, label]
    """
    def tokenizer(text):
        return [tok.lower() for tok in text.split(' ')]
    return [tokenizer(sentence) for sentence, _ in data]

# 根据分好词的训练数据集来创建词典
def get_vocab_imdb(data):
    tokenized_data = get_tokenized_imdb(data)
    counter = collections.Counter([tk for st in tokenized_data for tk in st]) # 计数,并生成一个字典
    return Vocab.Vocab(counter, min_freq=5) # 过滤掉了出现次数少于5的词
"""
TEXT.vocab.Vocab返回三个属性:
freqs 用来返回每一个单词和其对应的频数。
itos 按照下标的顺序返回每一个单词(一个列表(索引到词的映射))
stoi 返回每一个单词与其对应的下标(stoi:词到索引的字典)
"""

# 数据预处理
# 因为每条评论长度不一致所以不能直接组合成小批量,我们对每条评论进行分词之后
# 把词典转换成词索引,然后通过截断或者补0来将每条评论长度固定成500
def preprocess_imdb(data, vocab):
    max_l = 500  # 将每条评论通过截断或者补0，使得长度变成500
    def pad(x):
        if len(x) > max_l: 
            return x[:max_l] 
        else:
            return x + [0] * (max_l - len(x))
             
    tokenized_data = get_tokenized_imdb(data) # 按空格分词
    features = torch.tensor([pad([vocab.stoi[word] for word in sentence]) for sentence in tokenized_data])
    labels = torch.tensor([score for _, score in data])
    return features, labels

# 用这些词向量(嵌入向量)作为评论中每个词的特征向量。
# 注意,预训练词向量的维度需要与创建的模型中的嵌入层输出大小embed_size一致
def load_pretrained_embedding(words, pretrained_vocab):
    """从预训练好的vocab中提取出words对应的词向量"""
    embed = torch.zeros(len(words), pretrained_vocab.vectors[0].shape[0]) # 初始化为0
    oov_count = 0 # out of vocabulary
    for i, word in enumerate(words):
        try:
            idx = pretrained_vocab.stoi[word]
            embed[i, :] = pretrained_vocab.vectors[idx]
        except KeyError:
            oov_count += 1
    if oov_count > 0:
        print("There are %d oov words." % oov_count)
    return embed

# 定义预测函数
def predict_sentiment(net, vocab, sentence):
    device = list(net.parameters())[0].device
    sentence = torch.tensor([vocab.stoi[word] for word in sentence], device=device)
    label = torch.argmax(net(sentence.view((1, -1))), dim=1)
    return 'positive' if label.item() == 1 else 'negative'
