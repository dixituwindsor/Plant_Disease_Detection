from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.mobile_optimizer import optimize_for_mobile

from flask import Flask, request, jsonify

app = Flask(__name__)

classes = [ 'Apple___Apple_scab',
            'Apple___Black_rot',
            'Apple___Cedar_apple_rust',
            'Apple___healthy',
            'Blueberry___healthy',
            'Cherry_(including_sour)___Powdery_mildew',
            'Cherry_(including_sour)___healthy',
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
            'Corn_(maize)___Common_rust_',
            'Corn_(maize)___Northern_Leaf_Blight',
            'Corn_(maize)___healthy',
            'Grape___Black_rot',
            'Grape___Esca_(Black_Measles)',
            'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
            'Grape___healthy',
            'Orange___Haunglongbing_(Citrus_greening)',
            'Peach___Bacterial_spot',
            'Peach___healthy',
            'Pepper,_bell___Bacterial_spot',
            'Pepper,_bell___healthy',
            'Potato___Early_blight',
            'Potato___Late_blight',
            'Potato___healthy',
            'Raspberry___healthy',
            'Soybean___healthy',
            'Squash___Powdery_mildew',
            'Strawberry___Leaf_scorch',
            'Strawberry___healthy',
            'Tomato___Bacterial_spot',
            'Tomato___Early_blight',
            'Tomato___Late_blight',
            'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot',
            'Tomato___Spider_mites Two-spotted_spider_mite',
            'Tomato___Target_Spot',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
            'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy'  ]


class SimpleResidualBlock(nn.Module):
   def __init__(self):
      super().__init__()
      self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
      self.relu1 = nn.ReLU()
      self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
      self.relu2 = nn.ReLU()

   def forward(self, x):
      out = self.conv1(x)
      out = self.relu1(out)
      out = self.conv2(out)
      return self.relu2(out) + x  # ReLU can be applied before or after adding the input


def accuracy(outputs, labels):
   _, preds = torch.max(outputs, dim=1)
   return torch.tensor(torch.sum(preds==labels).item() / len(preds))


# base class for the model
class ImageClassificationBase(nn.Module):

   def training_step(self, batch):
      images, labels = batch
      out = self(images)  # Generate predictions
      loss = F.cross_entropy(out, labels)  # Calculate loss
      return loss

   def validation_step(self, batch):
      images, labels = batch
      out = self(images)  # Generate prediction
      loss = F.cross_entropy(out, labels)  # Calculate loss
      acc = accuracy(out, labels)  # Calculate accuracy
      return {"val_loss": loss.detach(), "val_accuracy": acc}

   def validation_epoch_end(self, outputs):
      batch_losses = [x["val_loss"] for x in outputs]
      batch_accuracy = [x["val_accuracy"] for x in outputs]
      epoch_loss = torch.stack(batch_losses).mean()  # Combine loss
      epoch_accuracy = torch.stack(batch_accuracy).mean()
      return {"val_loss": epoch_loss, "val_accuracy": epoch_accuracy}  # Combine accuracies

   def epoch_end(self, epoch, result):
      print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
         epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_accuracy']))


# convolution block with BatchNormalization
def ConvBlock(in_channels, out_channels, pool=False):
   layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
             nn.BatchNorm2d(out_channels),
             nn.ReLU(inplace=True)]
   if pool:
      layers.append(nn.MaxPool2d(4))
   return nn.Sequential(*layers)


# resnet architecture
class ResNet9(ImageClassificationBase):
   def __init__(self, in_channels, num_diseases):
      super().__init__()

      self.conv1 = ConvBlock(in_channels, 64)
      self.conv2 = ConvBlock(64, 128, pool=True)  # out_dim : 128 x 64 x 64
      self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))

      self.conv3 = ConvBlock(128, 256, pool=True)  # out_dim : 256 x 16 x 16
      self.conv4 = ConvBlock(256, 512, pool=True)  # out_dim : 512 x 4 x 44
      self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))

      self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                      nn.Flatten(),
                                      nn.Linear(512, num_diseases))

   def forward(self, xb):  # xb is the loaded batch
      out = self.conv1(xb)
      out = self.conv2(out)
      out = self.res1(out) + out
      out = self.conv3(out)
      out = self.conv4(out)
      out = self.res2(out) + out
      out = self.classifier(out)
      return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet9(3, 38).to(device)  # Move the model to the device
model.load_state_dict(torch.load('plant-disease-model.pth', map_location=device))
model.eval()

# Trace the model to generate TorchScript representation
example_input = torch.rand(1, 3, 256, 256, device=device)
traced_model = torch.jit.trace(model, example_input)
traced_model.save('plant-disease-model.pt')
optimized_scripted_module = optimize_for_mobile(traced_model)
optimized_scripted_module._save_for_lite_interpreter("plant-disease-model-mobile.ptl")


def predict_image(img_pil, model, device):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(img_tensor)
        _, predicted = torch.max(preds, 1)
        predicted_class = classes[predicted.item()]

    return predicted_class



@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        if file:
            try:
                img = Image.open(file)
                predicted_class = predict_image(img, model, device)
                return {"pred": predicted_class}
            except Exception as e:
                return jsonify({'pred': str(e)})
    else:
        return {"pred": "No image"}


if __name__ == '__main__':
    app.run(debug=True, host="192.168.203.105")