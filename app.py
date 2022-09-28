from flask import Flask
import torch
from flask import request
from flask import Flask, render_template
from flask_cors import CORS
from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import OrderedDict
from torchvision import datasets, transforms, models

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
def load_ckpt(ckpt_path):
    ckpt = torch.load(ckpt_path)

    model = models.resnet18(pretrained=True)
    model.fc = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(512, 400)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(400, 200)),
                          ('relu',nn.ReLU()),
                          ('fc3', nn.Linear(200, 3)),
                          ('output', nn.LogSoftmax(dim=1))
                      ]))

    model.load_state_dict(ckpt, strict=False)

    return model
modelz = load_ckpt('Race_classifier.pth')
import PIL
def process_image(image):
    #Processing the image so that it can be fed to the model created
    im = PIL.Image.open(image)
    return test_transforms(im)

def predict(image_path, model):
    model.eval()
    img_pros = process_image(image_path)
    img_pros = img_pros.view(1,3,224,224)
    with torch.no_grad():
        output = model(img_pros)
    return output

@app.route('/result',methods=["POST"])
def result():
    img = request.files['img']
    ps = torch.exp(predict(img, modelz))
    cls_score = int(torch.argmax(torch.exp(ps)))
    if cls_score == 0:
        return 'The model is '+str(ps[0][0]*100).replace('tensor','').replace(')','').replace('(','')+'%'+ ' sure that it is East Asian with chances of European being '+str(ps[0][1]*100).replace('tensor','').replace(')','').replace('(','')+'%'+" and South Asian being "+str(ps[0][2]*100).replace('tensor','').replace(')','').replace('(','')+'%' 
    elif cls_score==1:
        return 'The model is '+str(ps[0][1]*100).replace('tensor','').replace(')','').replace('(','')+'%'+ ' sure that it is European with chances of East Asian being '+str(ps[0][0]*100).replace('tensor','').replace(')','').replace('(','')+'%'+" and South Asian being "+str(ps[0][2]*100).replace('tensor','').replace(')','').replace('(','')+'%'
    else:
        return 'The model is '+str(ps[0][2]*100).replace('tensor','').replace(')','').replace('(','')+'%'+ ' sure that it is South Asian with chances of East Asian being '+str(ps[0][0]*100).replace('tensor','').replace(')','').replace('(','')+'%'+" and European "+str(ps[0][1]*100).replace('tensor','').replace(')','').replace('(','')+'%'

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)