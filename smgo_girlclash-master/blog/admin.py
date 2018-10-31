from django.contrib import admin
from .models import Post
from .models import Picture, Etc, Face, Fashion, Food, Nature, Pet
admin.site.register(Post)
admin.site.register(Picture)
admin.site.register(Face)
admin.site.register(Fashion)
admin.site.register(Food)
admin.site.register(Nature)
admin.site.register(Pet)
admin.site.register(Etc)