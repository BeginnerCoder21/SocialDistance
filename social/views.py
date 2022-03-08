from django.shortcuts import render
from django.http import HttpResponse
import cv2 as cv
from scipy.spatial import distance as dist
import numpy as np
import argparse  
import imutils 
import os  

def index(request):
    return render(request, 'index.html')
    # return render(request, 'index.html')
    # return HttpResponse('<h1>Hello hii</h1>')\

def download(request):
    import sociald
    inp_vid = request.GET['file']
    return render(request, 'download.html',{'video':inp_vid})
