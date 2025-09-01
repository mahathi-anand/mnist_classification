from header import * #All necessary packages in header file


#Loading the MNIST data
print_info("Loading the MNIST Training and Test Data")
save_path = os.path.join(os.path.dirname(__file__), "data")
data_train = MNIST(root = save_path, train = True, download = True)
data_test = MNIST(root = save_path, train = False, download = True)

print("Train set size:", len(data_train))
print("Test set size:", len(data_test))

#Plotting some data to check the data structure
print_info("Data Visualization")

fig, axes = plt.subplots(1,4, figsize = (10,3))
for i in range(4):
    data_point = data_point = np.random.randint(0,len(data_train)-1)
    image, label = data_train[data_point]
    axes[i].imshow(image, cmap = 'gray')
    axes[i].set_title(f'Label : {label}')
    axes[i].axis("off")

plt.tight_layout()
plt.show()

