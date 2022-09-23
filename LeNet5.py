
import numpy as np
import torch 
from torch import nn                        # 모델을 만들때 사용되는 base class 이며 클래스로 모델을 만들때는 nn, 함수로 바로 사용할때는 nn.functional을 사용한다. (편의에 따라 사용)
                                            # nn의 경우 weight를 사용하지만 nn.functional의 경우 filter를 사용하는 경우가 많다.(functional도 weight를 사용할 수 는 있다.) 
from torch.utils.data import DataLoader     # 데이터작업을 위한 기본요소 : dataloader - datasets를 순회 가능한 iterable 로 감싼다. datasets - 샘플과 정답을 저장한다. 
from torchvision import datasets            # torchvision의 datasets은 CIFAR, COCO등과 같은 비전 데이터를 갖고있다. (https://pytorch.org/vision/stable/datasets.html - 전체 데이터 셋)
from torchvision.transforms import ToTensor # TorchText, TorchVision, TorchAudio은 같이 도메인 특화 라이브러리로 데이터 셋을 제공한다. 
from torchvision import transforms
from torchviz import make_dot

print("===========================TRAINNING START===========================")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

batch_size = 64 # dataloader의 각 객체는 64개의 특징, 정답을 batch의 크기만큼 묶어 반환한다.

# MNIST 이미지 사이즈 조정.
data_Transform = transforms.Compose(            
    [
        transforms.Resize((32,32)),
        transforms.ToTensor(),
    ]
)

#MNIST 데이터셋 다운로드 및 Transform
train_dataset = datasets.MNIST( 
    root = "MINIST",
    train = True,
    download = True,
    transform = data_Transform,    
)


#MNIST 데이터셋 다운로드 및 Transform
test_dataset = datasets.MNIST( 
    root = "MINIST",
    train = False,
    download = True,
    transform = data_Transform,    
)

#MNIST_Data
test_dataloader = DataLoader(test_dataset, batch_size= batch_size, shuffle= True)
train_dataloader = DataLoader(train_dataset, batch_size= batch_size, shuffle= True)


for Image, Label in test_dataloader:                                     
    print(f"Shape of Image [N, C, H, W]: {Image.shape} {Image.dtype}")          
    print(f"Shape of Label: {Label.shape} {Label.dtype}")                       
    break

class LeNet(nn.Module) :
    def __init__(self):
        super(LeNet, self).__init__()  
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)          
        
    def forward(self, x):      
        x = nn.Tanh(self.conv1(x))
        x = nn.AvgPool2d(x, 2, 2)
        x = nn.Tanh(self.conv2(x))
        x = nn.AvgPool2d(x, 2, 2)
        x = nn.Tanh(self.conv3(x))
        x = x.view(-1 ,120)
        x = nn.Tanh(self.fc1(x))
        x = self.fc2(x)        
        return nn.Softmax(x, dim=1)

model = LeNet().to(device)
print("장치는 : {}, 모델은 : {}".format(device, model))

loss_fn = nn.CrossEntropyLoss()                             # 손실 함수 중 다중분류에 적합한 CrossEntropyLoss 함수 이다. 
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)    # 최적화 함수 중 확률적 경사하강법을 사용하여 데이터 수의 영향을 최소화한다.

def train(test_dataloader, model, loss_fn, optimizer):           # Dataloader: Dataset을 Iterate 하기 위함, Model: MyModel의 규칙, Loss_fn: 손실 함수, Optimizer : 최적화 함수
    size = len(test_dataloader.dataset)                          # Batch와 DataLoader에 올라간 Dataset의 사이즈를 곱하여 현재 학습이 진행된 양을 산출하는 역할
    
    for batch, (Image, Label) in enumerate(test_dataloader):             # 
        Image, Label = Image.to(device), Label.to(device)

        # 예측 오류 계산
        pred = model(Image)
        loss = loss_fn(pred, Label)

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(Image)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
            
def test(test_dataloader, model, loss_fn):
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for Image, Label in test_dataloader:
            Image, Label = Image.to(device), Label.to(device)
            pred = model(Image)
            test_loss += loss_fn(pred, Label).item()
            correct += (pred.argmax(1) == Label).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    
epochs = 1
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(test_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

torch.save(model.state_dict(), "mymodel.pth")
print("Saved PyTorch Model State to model.pth")

model = LeNet()
model.load_state_dict(torch.load("mymodel.pth"))

classes = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]

model.eval()
x, y = test_dataset[0][0], test_dataset[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')



'''

# 모델을 정의합니다.
class MyModel(nn.Module):                        # nn을 통한 MyModel을 생성.
    def __init__(self):
        super(MyModel, self).__init__()          # 파이썬은 클래스간 상속이 가능하다. super 명령어는 이러한 상속관계에서 부모 클래스를 호출하는 함수이다.
        
        self.flatten = nn.Flatten()              # __init__()에서 신경망의 계층(Layer)을 정의하고 forward에서 신경망에 데이터를 어떻게 전달할지 정한다.
        
        self.linear_relu_stack = nn.Sequential(  # nn.Sequential 클래스는 nn.Linear, nn.ReLU와 같은 모듈을 인수로 받아서 순서대로 정렬하고, 
                                                 #  입력값이 들어오면 순서에 따라 모듈을 실행하여 결과를 반환한다.
                                                 
            nn.Linear(28*28, 512),               # 입력되는 차원의 수와 출력되는 차원의 수를 인자값으로 정한다. 
            nn.ReLU(),                           # torch.nn.Linear(in_features, out_features, bias = True, device = None, dtype = None)
            nn.Linear(512, 512),                 # bias가 false로 설정된 경우 layer는 bias를 학습하지 않는다. 디폴트는 true 
            nn.ReLU(),                           # device는 cpu, gpu를 선택하는 것이다.   
            nn.Linear(512, 10)                   # dtype은 자료형을 선택하는 것이다. 
        )

    def forward(self, x):                       # 신경망에 데이터를 어떻게 전달할지 정한다.
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = MyModel().to(device) # CUDA가 usable할 경우 GPU를 기반으로 학습을 가속화 시키기 위하여 사용한다.
#model = MyModel()


'''
#모델 매개변수 최적화하기 
#매개변수 = 가중치(weight)를 의미한다. 
#모델을 학습 시키기 위하여 손실함수(Loss Function)과 최적화(Optimizer)가 필요하다. 
#최적화 : 확률적 경사하강법 사용. SGD - Stochastic Gradient Descent
'''


print("장치는 : {}, 모델은 : {}".format(device, model))

loss_fn = nn.CrossEntropyLoss()                             # 손실 함수 중 다중분류에 적합한 CrossEntropyLoss 함수 이다. 
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)    # 최적화 함수 중 확률적 경사하강법을 사용하여 데이터 수의 영향을 최소화한다.

def train(dataloader, model, loss_fn, optimizer):           # Dataloader: Dataset을 Iterate 하기 위함, Model: MyModel의 규칙, Loss_fn: 손실 함수, Optimizer : 최적화 함수
    size = len(dataloader.dataset)                          # Batch와 DataLoader에 올라간 Dataset의 사이즈를 곱하여 현재 학습이 진행된 양을 산출하는 역할
    
    for batch, (Image, Label) in enumerate(dataloader):             # 
        Image, Label = Image.to(device), Label.to(device)

        # 예측 오류 계산
        pred = model(Image)
        loss = loss_fn(pred, Label)

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(Image)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
            
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for Image, Label in dataloader:
            Image, Label = Image.to(device), Label.to(device)
            pred = model(Image)
            test_loss += loss_fn(pred, Label).item()
            correct += (pred.argmax(1) == Label).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    
epochs = 1
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

torch.save(model.state_dict(), "mymodel.pth")
print("Saved PyTorch Model State to model.pth")

model = MyModel()
model.load_state_dict(torch.load("mymodel.pth"))

classes = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
    '''
