import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Function

# ==========================================
# CẤU HÌNH
# ==========================================
class Args:

    batch_size = 16
    test_batch_size = 1000
    # Paper train 300 epochs để đạt hội tụ tốt nhất
    epochs = 400
    # Learning rate khởi điểm 1e-2 (0.01)
    lr = 0.01
    no_cuda = False
    seed = 1
    log_interval = 200

    num_sng = 4

args = Args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# ==========================================
# CÁC LỚP CỐT LÕI
# ==========================================
class UnipolarSte(Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()
    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)

class MPTSInputEncoding(nn.Module):
    def __init__(self, num_sng=8, threshold_ratio=0.75):
        super(MPTSInputEncoding, self).__init__()
        self.num_sng = num_sng
        self.threshold_ratio = threshold_ratio

    def forward(self, x):
        votes = torch.zeros_like(x)
        for _ in range(self.num_sng):
            noise = torch.rand_like(x)
            votes += (x > noise).float()
        threshold = self.num_sng * self.threshold_ratio
        return (votes >= threshold).float()

class TrainableThresholdActivation(nn.Module):
    def __init__(self, num_features, steepness=1.0): # Khởi tạo steepness thấp = 1.0
        super(TrainableThresholdActivation, self).__init__()
        self.tau = nn.Parameter(torch.zeros(num_features))
        self.steepness = steepness

    def forward(self, x):
        # Sigmoid với độ dốc thay đổi dần
        return UnipolarSte.apply(torch.sigmoid(self.steepness * (x - self.tau)) - 0.5)

class SBNLinear(nn.Module):
    def __init__(self, in_features, out_features, use_scaling=True):
        super(SBNLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.use_scaling = use_scaling
        if self.use_scaling:
            self.scaling_factor = nn.Parameter(torch.ones(out_features))
        else:
            self.register_parameter('scaling_factor', None)

    def forward(self, x):
        w_b = UnipolarSte.apply(self.weight)
        output = F.linear(x, w_b)
        if self.use_scaling:
            s_eff = self.scaling_factor.clamp(min=1).round().detach() - self.scaling_factor.detach() + self.scaling_factor
            output = output * s_eff.unsqueeze(0)
        return output

# ==========================================
# MÔ HÌNH FC-256
# ==========================================
class SBN_SF_Net_Optimized(nn.Module):
    def __init__(self):
        super(SBN_SF_Net_Optimized, self).__init__()

        # Tăng chất lượng đầu vào với 4 SNG
        self.input_encoding = MPTSInputEncoding(num_sng=args.num_sng, threshold_ratio=0.75)

        # Layer 1: 784 -> 256
        self.fc1 = SBNLinear(784, 256)
        self.act1 = TrainableThresholdActivation(256, steepness=1.0)

        # Output Layer: 256 -> 10
        self.fc4 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.input_encoding(x)

        x = self.fc1(x)
        x = self.act1(x)

        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

    # Hàm tăng độ cứng của Sigmoid theo thời gian (Annealing)
    def update_steepness(self, epoch):
        # Tăng dần từ 1 lên 9
        # Giúp gradient mượt ở đầu (dễ học) và cứng ở cuối (chuẩn Binary)
        new_s = min(9.0, 1.0 + (epoch / 30.0))
        self.act1.steepness = new_s
        return new_s

# ==========================================
# KHỞI TẠO & TRAINING
# ==========================================

kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([transforms.ToTensor()])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([transforms.ToTensor()])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

model = SBN_SF_Net_Optimized()
if args.cuda:
    model.cuda()

# 1. Data-Aware Initialization (Bắt buộc cho No-BatchNorm)
def data_aware_init(model, loader, device):
    print("\n--- Data-Aware Initialization ---")
    model.eval()
    data, _ = next(iter(loader))
    data = data.to(device)
    with torch.no_grad():
        x = data.view(-1, 784)
        x = model.input_encoding(x)
        out1 = model.fc1(x)
        mean1 = out1.mean(dim=0)
        model.act1.tau.data.copy_(mean1)
        print(f"Layer 1 Tau Mean Init: {mean1.mean().item():.2f}")
    print("Khởi tạo hoàn tất!\n")
    model.train()

data_aware_init(model, train_loader, torch.device("cuda" if args.cuda else "cpu"))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# 2. Learning Rate Scheduler
# 1e-2 (0-100), 1e-3 (100-200), 1e-4 (200-300)
def adjust_learning_rate(optimizer, epoch):
    lr = args.lr
    if epoch > 100:
        lr = args.lr * 0.1
    if epoch > 200:
        lr = args.lr * 0.01
    if epoch > 300:
        lr = args.lr * 0.001
    if epoch > 400:
        lr = args.lr * 0.0001
    if epoch > 450:
        lr = args.lr * 0.00001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(epoch):
    model.train()
    current_steepness = model.update_steepness(epoch)

    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda: data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
        optimizer.step()

    # Log 1 lần mỗi epoch cho gọn
    print(f'Epoch: {epoch} | Loss: {loss.item():.4f} | Steepness: {current_steepness:.1f}')

def test():
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda: data, target = data.cuda(), target.cuda()
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    acc = 100. * correct / len(test_loader.dataset)
    print(f'>>> Test Accuracy: {acc:.2f}%')
    return acc

print("Bắt đầu huấn luyện tối ưu hóa (300 Epochs)...")
best_acc = 0
for epoch in range(1, args.epochs + 1):
    adjust_learning_rate(optimizer, epoch)
    train(epoch)
    acc = test()

    # Lưu checkpoint tốt nhất
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "mnist_sbn_1fc256__4sng_learning_rate_0_9_best_16_400.pth")

print(f"\nĐã hoàn thành! Độ chính xác cao nhất đạt được: {best_acc:.2f}%")