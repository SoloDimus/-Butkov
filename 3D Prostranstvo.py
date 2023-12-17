# importing mplot3d toolkits
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import json
import random
from dataset import *
from nn imp

joper = Net()
joper = nn.DataParallel(joper,[device])
joper = joper.to(device)

PATH = Path("weights")
optimizer = optim.Adam(joper.parameters(), lr=0.001)
checkpoint = torch.load(PATH)
model_dict = checkpoint['model_state_dict']
joper.load_state_dict(model_dict)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

path = None if len(sys.argv) < 2 else sys.argv[1]

fig = plt.figure()

# syntax for 3-D projection
ax = plt.axes(projection ='3d')
if "dataset.json" not in os.listdir(Path(".")):
    if not path:
        path = input("dataset.json does not exitst in this directory, please type path to folder with solution -> ")

    json_streaming_writer(Path("dataset.json"), get_dataset, [Path(path)])

with open(Path(".") / Path("dataset.json"), 'r') as json_file:
    data = random.choices(json.load(json_file),k=10000)

x=[]
y=[]
z=[]
c=[]
def to_rgb(arr):
    arr = list(arr)
    for y,i in enumerate(arr):
        arr[y] = [i, 0, 1 - i]
    return arr
def normalize(arr):
    arr = np.array(arr)
    MIN = min(arr)
    arr += MIN
    MAX = max(arr)
    arr /= MAX
    return to_rgb(arr)
for i in data:
    pos = i['CentrePosition']
    x.append(pos["X"])
    y.append(pos["Z"])
    z.append(pos["Y"])
    c.append(joper(torch.tensor([pos["X"],pos["Y"],pos["Z"],cos(i["Angle"]),sin(i["Angle"]),i["Velocity"]],device=device)).item())
ax.scatter(x, y, z,c = c)
ax.set_title('3d Scatter plot geeks for geeks')
plt.show()