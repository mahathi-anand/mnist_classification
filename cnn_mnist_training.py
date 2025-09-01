from header import * #All necessary packages in header file
from modules import *


#Seeds for reproducibility
np.random.seed(0)

#Loading the transformed MNIST data 

transform = transforms.Compose([transforms.ToTensor()])

print_info("Loading the MNIST Training and Test Data")
save_path = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(save_path, exist_ok=True)
data_train = MNIST(root = save_path, train = True, transform = transforms.ToTensor(), download = False)
data_test = MNIST(root = save_path, train = False, transform = transforms.ToTensor(), download = False)

#Splitting test data for cross validation
size_val = int(len(data_test)/2)
size_test = len(data_test) - size_val


#Diving into features and labels
data_val, data_dest = random_split(data_test, [size_val, size_test])
train_x, train_y = process_data(data_train)
val_x, val_y = process_data(data_val)
test_x, test_y = process_data(data_test)

#Check for CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"

#Model and Data Instantiation

model = cnn_model()
model = model.to(device)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = model.parameters(), lr = 0.01)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)

dataset = TensorDataset(train_x, train_y)
batch_size = 128
data_loader = DataLoader(dataset, batch_size = batch_size, shuffle = True)

#Defining accuracy metrics required for cross-validation during training

num_classes = 10
train_acc = Accuracy(num_classes= num_classes, task = "multiclass")
train_prec = Precision(num_classes= num_classes, task = "multiclass")
train_recall = Recall(num_classes= num_classes, task = "multiclass")

val_acc = Accuracy(num_classes= num_classes, task = "multiclass")
val_prec = Precision(num_classes= num_classes, task = "multiclass")
val_recall = Recall(num_classes= num_classes, task = "multiclass")

#Training the model
print_info("Training the Model")
train_model(model, data_loader, val_x, val_y, 10, optimizer, loss,
            device, train_acc, train_prec, train_recall, val_acc, val_prec, val_recall)


print_info("Training Complete...Validating the Model")

#Testing and Visualizing the Predictions
test_acc = Accuracy(num_classes = num_classes, task = "multiclass")
test_prec = Accuracy(num_classes = num_classes, task = "multiclass")
test_recall = Accuracy(num_classes = num_classes, task = "multiclass")

model.eval()

with torch.inference_mode():
    y_pred = model(test_x.unsqueeze(dim = 1)).squeeze()
    pred_classes = torch.argmax(y_pred, dim=1)
    test_acc.update(pred_classes, test_y)
    test_prec.update(pred_classes, test_y)
    test_recall.update(pred_classes, test_y)

    acc= test_acc.compute()
    prec= test_prec.compute()
    recall=test_recall.compute()
    F1_score = 2*prec*recall/(prec+recall)

print(f"""Test Accuracy: {acc} | Test Precision: {prec} \n
        Test Recall: {recall} |  Test F1Score: {F1_score}""")

#Plotting a couple of random examples for visualization along with labels

fig, axes = plt.subplots(1,4, figsize = (10,3))
for i in range(4):
    data_point = np.random.randint(0, len(data_test))
    number = data_test[data_point][0]
    pred = model(number.unsqueeze(dim = 1)).squeeze()
    pred_label = torch.argmax(pred)
    axes[i].imshow(number.squeeze()  , cmap='gray')
    axes[i].set_title(f"Classified label: {pred_label}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()
