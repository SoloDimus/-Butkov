from nn import *

import torch.nn as nn

# path = input("Dataset does not exitst in this directory, please type path to folder with solutions -> ") if len(sys.argv) < 2 else sys.argv[1]

if len(sys.argv) == 2:
    with open(Path(sys.argv[1]),"r+") as file:
        data = [tuple(map(float, i.split())) for i in file] # X Y Z ANGLE VELOCITY
elif len(sys.argv) == 6:
    data = [[float(i) for i in sys.argv[1:]]]
else:
    print("Usage: python net.py 'filename' \nOr python net.py X Y Z ANGLE(RAD) VELOCITY(M/S)")
    exit(1)

joper = Net()
joper = nn.DataParallel(joper,[device])
joper = joper.to(device)

PATH = Path("weights")
optimizer = optim.Adam(joper.parameters(), lr=0.001)
checkpoint = torch.load(PATH)
# model_dict = {k.replace("module.",""):v for k, v in checkpoint['model_state_dict'].items()}
model_dict = checkpoint['model_state_dict']
joper.load_state_dict(model_dict)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

correct = 0
total = 0
errors = set()
outputs = []
for i in data:
    x_data = torch.tensor([i[0],i[1],i[2],cos(i[3]),sin(i[3]),i[4]],device=device)
    outputs.append(joper(x_data).item())
print(*outputs, ' ')
        # print(output)
    # if total % 3000 == 0:
        # print(total//3000,'%')
# print("accuracy", round(sum(errors).item()/len(errors), 3))
# print("precision", round(correct/total, 3))