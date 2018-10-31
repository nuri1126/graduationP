from django import forms

from .models import Post

class PostForm(forms.ModelForm):

    class Meta:
        model = Post
        fields = ('title', 'text','image')

class your_form_name(forms.Form):
      error_css_class = 'error' #custom css for form errors - ".error";
      required_css_class = 'required'#custom css for required fields - ".required";
      info_text = forms.CharField()
