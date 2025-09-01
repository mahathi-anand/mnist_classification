from header import * #All necessary packages in header file

#Necessary functions for training

def process_data(data_set):
    loader = DataLoader(data_set, batch_size = len(data_set))
    data_features, data_target = next(iter(loader))

    data_features = data_features.reshape(len(data_set),28,28)
    data_target = data_target.long().squeeze()
    #data_target = one_hot(data_target.long(),10).squeeze()

    return data_features, data_target

def cnn_model():
    model = nn.Sequential(
        nn.ZeroPad2d(2),
        nn.Conv2d(1,16,5,1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.LazyLinear(out_features = 10)
        )
    return model

def train_model(model, data_loader, val_x, val_y, epochs, optimizer, loss, device, train_acc, train_prec, train_recall, val_acc, val_prec, val_recall):
    
    for epoch in range(epochs):
        print("Epoch:", epoch)
        model.train()

        epoch_train_loss = 0
        epoch_val_loss = 0

        for batch in data_loader:
            inputs, outputs = batch
            inputs, outputs = inputs.to(device), outputs.to(dtype=torch.long, device=device)

            #Forward Pass
            y_pred = model(inputs.unsqueeze(dim = 1)).squeeze()

            train_loss = loss(y_pred, outputs)
            epoch_train_loss += train_loss.item()

            #Update Metrics
            pred_classes = torch.argmax(y_pred,dim=1)
            train_acc.update(pred_classes, outputs)
            train_prec.update(pred_classes, outputs)
            train_recall.update(pred_classes, outputs)

            #Update model
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            #Testing and cross validation
            model.eval()
            with torch.inference_mode():
                #Forward Pass
                pred_val = model(val_x.unsqueeze(dim=1)).squeeze()
                pred_loss = loss(pred_val, val_y.long())
                epoch_val_loss += pred_loss.item()
                pred_classes= torch.argmax(pred_val, dim=1)
                val_acc.update(pred_classes, val_y)
                val_prec.update(pred_classes, val_y)
                val_recall.update(pred_classes, val_y)

        epoch_train_acc = train_acc.compute()
        epoch_train_prec = train_prec.compute()
        epoch_train_recall = train_recall.compute()

        epoch_val_acc = val_acc.compute()
        epoch_val_prec = val_prec.compute()
        epoch_val_recall = val_recall.compute()

        print(f"""Epoch: {epoch} | Train Loss: {epoch_train_loss:.4f} | Validation loss: {epoch_val_loss:.4f}, 
              \n Train Accuracy: {epoch_train_acc.item():.4f}, Train Precision: {epoch_train_prec.item():.4f}, Train Recall: {epoch_train_recall.item():.4f},
              \n Validation Accuracy: {epoch_val_acc:.4f}, CV Precision: {epoch_val_prec:.4f}, CV Recall: {epoch_val_recall:.4f}\n\n""")

        # Reset metrics for the next epoch
        train_acc.reset()
        train_prec.reset()
        train_recall.reset()

        val_acc.reset()
        val_prec.reset()
        val_recall.reset()











