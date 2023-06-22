from django.shortcuts import render,redirect
from django.http import HttpResponse
from.forms import RegistrationForm
from django.contrib.auth.views import LoginView
from django.urls import reverse_lazy
from gradio_client import Client
import tempfile
import json
import base64
from django.contrib.auth import login
import io
import os


# Create your views here.

def register(request):
    if request.method == 'POST':
        form = RegistrationForm(request.POST)
        if form.is_valid():
            user=form.save()
            login(request,user)
            return render(request,'base/registered.html')
        else:
            return HttpResponse("either username or password is not in according to policy")
            

    else:
        form = RegistrationForm()
        args = {'form':form}

        return render(request,'base/register.html',args)

class CustomizeLoginView(LoginView):
    template_name = 'base/login.html'
    fields = '__all__'
    redirect_authenticated_user = True

    def get_success_url(self):
        return reverse_lazy('home')



















def prediction_xception(image):
    client = Client("https://mista4444-plant-leaf-disease-detection.hf.space/")
    result = client.predict(
				image,	# str (filepath or URL to image) in 'img' Image component
				api_name="/predict" )
    return result


def prediction_densenet(image):
    client = Client("https://mista4444-densenet.hf.space/")
    result = client.predict(
				image,	# str (filepath or URL to image) in 'img' Image component
				api_name="/predict" )
    return result

def prediction_googlenet(image):
    client = Client("https://mista4444-googlenet.hf.space/")
    result = client.predict(
				image,	# str (filepath or URL to image) in 'img' Image component
				api_name="/predict")
    return result

def prediction_alexnet(image):
    client = Client("https://mista4444-alexnet.hf.space/")
    result = client.predict(
				image,	# str (filepath or URL to image) in 'img' Image component
				api_name="/predict")
    return result

def prediction_resnet(image):
    client = Client("https://mista4444-resnet.hf.space/")
    result = client.predict(
				image,	# str (filepath or URL to image) in 'img' Image component
				api_name="/predict")
    return result

def home(request):
    
    xception_link = "https://huggingface.co/spaces/mista4444/plant-leaf-disease-detection"
    densenet_link = "https://huggingface.co/spaces/mista4444/densenet"
    googlenet_link = "https://huggingface.co/spaces/mista4444/googlenet"
    alexnet_link = "https://huggingface.co/spaces/mista4444/alexnet"
    resnet_link = "https://huggingface.co/spaces/mista4444/resnet"

    context = {
        'xception': xception_link,
        'densenet': densenet_link,
        'googlenet': googlenet_link,
        'alexnet': alexnet_link,
        'resnet': resnet_link ,
    }



    if request.method == 'POST':
        
        

        photo = request.FILES['file']
        if 'PYTHONANYWHERE_DOMAIN' in os.environ:
            temp_file_path = "/tmp/" + photo.name
            with open(temp_file_path, 'wb') as temp_file:
                temp_file.write(photo.read())
            image_path = temp_file_path
        else:
        # Running locally
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(photo.read())
                temp_file_path = temp_file.name
            image_path = temp_file_path

        
    # Pass the temporary file path to the prediction functions
        pred_xception = prediction_xception(image_path)
        pred_densenet = prediction_densenet(image_path)
        pred_googlenet = prediction_googlenet(image_path)
        pred_alexnet = prediction_alexnet(image_path)
        pred_resnet = prediction_resnet(image_path)

        

        # Load the JSON file
        with open(pred_xception) as json_file:
            data_xception = json.load(json_file)

        label_x = data_xception['label']
        confidence_x = round((data_xception['confidences'][0]['confidence'])*100,2)

        context_x = {'label': label_x,
                     'confidence': confidence_x}

        with open(pred_densenet) as json_file:
            data_densenet= json.load(json_file)

        label_d = data_densenet['label']
        confidence_d = round((data_densenet['confidences'][0]['confidence'])*100,2)

        context_d = {'label': label_d,
                     'confidence': confidence_d}




        with open(pred_googlenet) as json_file:
            data_googlenet= json.load(json_file)
        
        label_g = data_googlenet['label']
        confidence_g = round((data_googlenet['confidences'][0]['confidence'])*100,2)

        context_g = {'label': label_g,
                     'confidence': confidence_g}



    

        with open(pred_alexnet) as json_file:
            data_alexnet= json.load(json_file)
        
        label_a = data_alexnet['label']
        confidence_a = round((data_alexnet['confidences'][0]['confidence'])*100,2)

        context_a = {'label': label_a,
                     'confidence': confidence_a}

    





        with open(pred_resnet) as json_file:
            data_resnet = json.load(json_file)

        label_r = data_resnet['label']
        confidence_r = round((data_resnet['confidences'][0]['confidence'])*100,2)

        context_r = {'label': label_r,
                     'confidence': confidence_r}








        

        return render(request,"base/result.html",{"context_x": context_x,
                                                "context_d":context_d,
                                                "context_g":context_g,
                                                "context_a":context_a,
                                                "context_r":context_r,
                                                })


        
  

    return render(request, 'base/home.html',context)