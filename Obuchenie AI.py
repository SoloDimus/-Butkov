from nn import *
from dataset import *

logger.info("Opening dataset file")

if "dataset" not in os.listdir(Path(".")):
    path = input("Dataset does not exitst in this directory, please type path to folder with solutions -> ") if len(sys.argv) < 2 else sys.argv[1]
    raw_streaming_writer("dataset", get_datasets, [Path(path)])


with open(Path("dataset"), 'r') as file:
    raw_data = [tuple(map(float, i.split())) for i in file]

random.shuffle(raw_data)

num_of_tests = len(raw_data) // 20

train = torch.tensor(raw_data[:-num_of_tests])
test = torch.tensor(raw_data[-num_of_tests:])

train = train.to(device)
test = test.to(device)

joper = Net()
joper = nn.DataParallel(joper,[device])
joper = joper.to(device)

optimizer = optim.Adam(joper.parameters(), lr=0.001)

logger.info("Start training")
k=0
numepoch = 1
numtrains = len(train)
for epoch in range(numepoch): 
    logger.info(f"Started {epoch}")
    for data in train:
        x_data, y_data = unpack_data(data)
        joper.zero_grad()  
        output = joper(x_data) 
        loss = F.mse_loss(output, y_data)  
        loss.backward() 
        optimizer.step() 
        k+=1
        if k % (numtrains // 100) == 0:
            logger.info(f"{k*100 // numtrains} %")

logger.info("End training")


path = Path(".") / Path("weights")
logger.info("Saving model")
try:
    torch.save({
                'model' : Net(),
                'epoch': epoch,
                'model_state_dict': joper.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, path)
    logger.info("Model saved")
except Exception as e:
    logger.error(f"Exeption occuried while saving: {e}")


correct = 0
total = 0
errors = set()
try:
    with torch.no_grad():
        for data in test:
            x_data, y_data = unpack_data(data)
            output = joper(x_data)
            if abs(output[0] - y_data[0]) < y_data[0]*0.05:
                correct += 1
            total += 1
            errors.add(abs(output[0] - y_data[0]))
except Exception as e:
    logger.error(f"Exeption occuried while testing: {e}")
logger.info(f"precision {round(correct/total, 3)}")
logger.info(f"accuracy {round(sum(errors).item()/len(errors), 3)}")
