from utils import *
from torchvision import models


model = GRU_linear()


total = sum([param.nelement() for param in model.parameters()])

print(total)