# views.py in my_app
from django.shortcuts import render

def web(request):

    return render(request, 'test.html')
