from django.shortcuts import render, get_object_or_404
from django.utils import timezone
from .models import Post
from .forms import PostForm
from django.shortcuts import redirect


#fileupload
import json

from django.http import HttpResponse
from django.views.generic import CreateView, DeleteView, ListView
from .models import Picture, Etc, Face, Fashion,Food, Nature,Pet
from .response import JSONResponse, response_mimetype
from .serialize import serialize

# Create your views here.
def post_list(request):
    posts = Post.objects.filter(published_date__lte=timezone.now()).order_by('created_date')
    pictures = Picture.objects.all()
    etcs = Etc.objects.all()
    faces = Face.objects.all()
    fashions = Fashion.objects.all()
    foods = Food.objects.all()
    natures = Nature.objects.all()
    pets = Pet.objects.all()

    params = {'pictures': pictures,'etcs':etcs,'faces':faces,'fashions':fashions,'foods':foods,'natures':natures,'pets':pets, 'posts': posts}
    return render(request, 'blog/post_list.html', params)

def post_detail(request, pk):
        post = get_object_or_404(Post, pk=pk)
        return render(request, 'blog/post_detail.html', {'post': post})

def post_new(request):
    if request.method == "POST":
        form = PostForm(request.POST, request.FILES)
        if form.is_valid():
            post = form.save(commit=False)
            post.author = request.user
            post.published_date = timezone.now()
            post.save()
            return redirect('post_detail', pk=post.pk)
    else:
        form = PostForm()
    return render(request, 'blog/post_edit.html', {'form': form})

def post_edit(request, pk):
    post = get_object_or_404(Post, pk=pk)
    if request.method == "POST":
        form = PostForm(request.POST, instance=post)
        if form.is_valid():
            post = form.save(commit=False)
            post.author = request.user
            post.published_date = timezone.now()
            post.save()
            return redirect('post_detail', pk=post.pk)
    else:
        form = PostForm(instance=post)
    return render(request, 'blog/post_edit.html', {'form': form})

# Point
from slim import testImage

def classify(request):
    if request.method == "GET":
        testImage.Classifier()
    return render(request, 'blog/picture_angular_form.html')

import os

# def view_photos(request):
#     images
#     return render(request, 'blog/base.html', {'images': images})
    # if request.method == "GET":
    #     images = get_object_or_404(Picture, pk=pk)
    #     # root = os.path.join(dataset_dir, 'pictures')
    #     # for filename in os.listdir(root):
    #     #     path = os.path.join(root, filename)
    #     #     #images.append(path)
    #     #     images.path = path
    # return render(request, 'blog/base.html', {'images': images})


#fileupload
class PictureCreateView(CreateView):
    model = Picture
    fields = "__all__"

    def form_valid(self, form):
        self.object = form.save()
        files = [serialize(self.object)]
        data = {'files': files}
        response = JSONResponse(data, mimetype=response_mimetype(self.request))
        response['Content-Disposition'] = 'inline; filename=files.json'
        return response

    def form_invalid(self, form):
        data = json.dumps(form.errors)
        return HttpResponse(content=data, status=400, content_type='application/json')

class BasicVersionCreateView(PictureCreateView):
    template_name_suffix = '_basic_form'


class BasicPlusVersionCreateView(PictureCreateView):
    template_name_suffix = '_basicplus_form'


class AngularVersionCreateView(PictureCreateView):
    template_name_suffix = '_angular_form'


class jQueryVersionCreateView(PictureCreateView):
    template_name_suffix = '_jquery_form'


class PictureDeleteView(DeleteView):
    model = Picture

    def delete(self, request, *args, **kwargs):
        self.object = self.get_object()
        self.object.delete()
        response = JSONResponse(True, mimetype=response_mimetype(request))
        response['Content-Disposition'] = 'inline; filename=files.json'
        return response


class PictureListView(ListView):
    model = Picture

    def render_to_response(self, context, **response_kwargs):
        files = [ serialize(p) for p in self.get_queryset() ]
        data = {'files': files}
        response = JSONResponse(data, mimetype=response_mimetype(self.request))
        response['Content-Disposition'] = 'inline; filename=files.json'
        return response

# def whatever(request):
#     if request.method == 'POST':
#         import subprocess
#         output = subprocess.check_output(["script.py", "--", request.POST['hiya']])  #executing hyojeong's first file extends script.py
#         return HttpResponse(output, content_type='text/plain')


import subprocess
from .forms import your_form_name

def your_view_name(request):
  if request.method == 'POST':
    form = your_form_name()
  else:
    if form.is_valid():
      info = request.POST['info_name']
      output = script_function(info)
        # Here you are calling script_function,
      # passing the POST data for 'info' to it;
      #return render(request, 'your_app/your_template.h

      return render(request, 'blog/your_template.html', {
        'info': info,
        'output': output,
      })

  #return render(request, 'your_app/your_template.html', {
    return render(request, 'blog/your_template.html', {
        'form': form,
    })

def script_function( post_from_form ):
    print (post_from_form); #optional,check what the function received from the submit;
    return subprocess.check_call(['/path/to/your/script.py', post_from_form])
