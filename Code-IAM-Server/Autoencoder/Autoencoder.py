
import torch
from matplotlib import pyplot as plt
import numpy as np
import SSIM
import pandas as pd
import os
from skimage import io
import glob
import torchvision
import copy
import glob

model_name = "model_epoch_59.model"
should_train = True    
should_test = True
save_images = True

class AE(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_input_laye1 = torch.nn.Conv2d(3,32,3, padding=1)
        self.encoder_hidden_layer2 = torch.nn.Conv2d(32,64,3, stride=2, padding=1)
        self.encoder_hidden_layer3 = torch.nn.Conv2d(64,128,3, stride=2, padding=1)
        self.encoder_hidden_layer4 = torch.nn.Conv2d(128,256,3, stride=2, padding=1)
        self.decoder_hidden_layer0 = torch.nn.ConvTranspose2d(256,128,3, stride=2, padding=1)
        self.decoder_hidden_layer1 = torch.nn.ConvTranspose2d(128,64,3, stride=2, padding=1)
        self.decoder_hidden_layer2 = torch.nn.ConvTranspose2d(64,32,3, stride=2, padding=1)
        self.decoder_output_layer = torch.nn.ConvTranspose2d(32,3,3, padding=1)

    def forward(self, features):
        activation = self.encoder_input_laye1(features)
        activation = torch.nn.functional.leaky_relu(activation)
        #print("e1",activation.shape)
        activation = self.encoder_hidden_layer2(activation)
        activation = torch.nn.functional.leaky_relu(activation)
        #print("e2",activation.shape)
        activation = self.encoder_hidden_layer3(activation)
        activation = torch.nn.functional.leaky_relu(activation)
        #print("e3",activation.shape)
        activation = self.encoder_hidden_layer4(activation)
        activation = torch.nn.functional.leaky_relu(activation)
        
        sze_enc=activation.shape[-1]
        activation = self.decoder_hidden_layer0(activation,output_size=(sze_enc*2,sze_enc*2))
        activation = torch.nn.functional.leaky_relu(activation)
        activation = self.decoder_hidden_layer1(activation,output_size=(sze_enc*4,sze_enc*4))
        activation = torch.nn.functional.leaky_relu(activation)
        #print("d1",activation.shape)
        activation = self.decoder_hidden_layer2(activation,output_size=(sze_enc*8,sze_enc*8))
        activation = torch.nn.functional.leaky_relu(activation)
        #print("d2",activation.shape)
        activation = self.decoder_output_layer(activation)
        #activation = torch.sigmoid(activation)
        #print("d3",activation.shape)
        return activation

class CancerDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, isTrain=True, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #self.frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.paths = []
        files = []
        neg_files = sorted(glob.glob(root_dir+"*/*_0_color.png"))
        pos_files = sorted(glob.glob(root_dir+"*/*_1_color.png"))
        percentile_80th = int(len(pos_files)*0.8)
        if isTrain == True:
            files = pos_files+neg_files#pos_files[:percentile_80th]
            #print("TRAIN files")
            #print(files)
        else:
            files = pos_files[percentile_80th:]+neg_files
            #print("TEST files")
            #print(files)
        #print("AAA",len(folders))
        #for file in files:
        self.paths = files
        self.paths = self.paths[::250]
        

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.paths[idx]#os.path.join(self.root_dir,self.paths[idx])
        image = io.imread(img_name)[::2,::2,:3]
        #plt.imshow(image)
        #image = cv2.resize(image, (256,256))
        #sample = {'image': image}

        if self.transform:
            image = self.transform(image)

        return image


if __name__ == "__main__":
    #  use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    criterion = SSIM.NegSSIM()
    
    '''train_dataset = torchvision.datasets.Flowers102(root='flowers', 
                                        split='train', 
                                        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Resize((256, 256))]), 
                                        download=True)'''
    train_dataset = CancerDataset("D:/CRC_Autoencoder/windows_2/", isTrain=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Resize((256, 256))]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    
    test_dataset = CancerDataset("D:/CRC_Autoencoder/windows_2/", isTrain=False, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Resize((256, 256))]))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=0)
    losses = []
    from EarlyStopping import * 
    early_stopping = EarlyStopping(warm_up=60, patience=10)
    
    if should_train:
        epochs = 60
        #model.load_state_dict(torch.load(model_name))
        
        for epoch in range(epochs):
            loss = 0    
            for index, batch_features in enumerate(train_loader):
                batch_features = batch_features.to(device)
                optimizer.zero_grad()
                #print(batch_features)
                #features, = batch_features
                
                outputs = model(batch_features)
                
                train_loss = criterion(outputs, batch_features)
                train_loss.backward()
                optimizer.step()
                loss += train_loss.item()
                print(f"\rEpoch {epoch}/{epochs}, iteration {index}/{len(train_loader)}. Average loss: {loss/(index+1)}", end="")
        
            loss = loss / len(train_loader)
            losses.append(loss)
            print("\nepoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))
            torch.save(model.state_dict(), f"model_all_epoch_{epoch}.model")
            if early_stopping(epoch+1, losses[epoch], copy.deepcopy(model)):   
                print('Early Stop at' + str(epoch))
                break
        torch.save(model.state_dict(), "model_all.model")
        print(losses)
        plt.plot(losses)
        plt.show()
    if should_test:
        model.load_state_dict(torch.load(model_name))#_epoch_3
        loss = 0
        for batch_features in test_loader:
            batch_features = batch_features.to(device)
            outputs = model(batch_features)
            if save_images:
                for i in range(outputs.shape[0]):
                    original_image = np.transpose(batch_features[i].detach().numpy(),(1,2,0))
                    
                    reconstructed_image = np.uint8(np.transpose(outputs[i].detach().numpy(),(1,2,0))*255)
                    plt.imshow(reconstructed_image)
                    plt.show()
                    print(f"\r{i}", end="")
                    plt.imsave(f"../Results/Images/{i}_original.png", np.uint8(original_image*255))
                    plt.imsave(f"../Results/Images/{i}_reconstructed.png", reconstructed_image)
                break
            it_loss = criterion(outputs, batch_features).item()
            print(f"\rTest Average loss: {it_loss:.2f}", end="")
            loss += it_loss
            break
        loss = loss / len(train_loader)
        print(f"\rTest loss: {loss}", end="")