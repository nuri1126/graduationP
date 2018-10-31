from django.conf.urls import include, url
from . import views

#fileupload
from django.http import HttpResponseRedirect
from django.conf import settings

# Uncomment the next two lines to enable the admin:
from django.contrib import admin
admin.autodiscover()

from blog.views import (
        BasicVersionCreateView, BasicPlusVersionCreateView,
        jQueryVersionCreateView, AngularVersionCreateView,
        PictureCreateView, PictureDeleteView, PictureListView,
        )

urlpatterns = [
    url(r'^slim/$', views.classify, name='classify'), # Point
    # url(r'^$', views.view_photos, name='view_photos'),
    url(r'^$', views.post_list, name='post_list'),
    url(r'^post/(?P<pk>[0-9]+)/$', views.post_detail, name='post_detail'),
    url(r'^post/new/$', views.post_new, name='post_new'),
    url(r'^post/(?P<pk>[0-9]+)/edit/$', views.post_edit, name='post_edit'),
#    url(r'^blabla$', views.whatever, name='hello'),
    url(r'^test$', views. your_view_name, name=' your_view_name'),


#fileupload
    url(r'^$', lambda x: HttpResponseRedirect('/upload/new/')),
    url(r'^admin/', include(admin.site.urls)),



    url(r'^upload/basic/$', BasicVersionCreateView.as_view(), name='upload-basic'),
    url(r'^upload/basic/plus/$', BasicPlusVersionCreateView.as_view(), name='upload-basic-plus'),
    url(r'^upload/new/$', PictureCreateView.as_view(), name='upload-new'),
    url(r'^upload/angular/$', AngularVersionCreateView.as_view(), name='upload-angular'),
    url(r'^upload/jquery-ui/$', jQueryVersionCreateView.as_view(), name='upload-jquery'),
    url(r'^upload/delete/(?P<pk>\d+)$', PictureDeleteView.as_view(), name='upload-delete'),
    url(r'^upload/view/$', PictureListView.as_view(), name='upload-view'),
]
if settings.DEBUG:
    from django.conf.urls.static import static
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
