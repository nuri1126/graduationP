from django.db import models
from django.utils import timezone
from mysite import settings

class Post(models.Model):
    author = models.ForeignKey('auth.User')
    title = models.CharField(max_length=200)
    text = models.TextField()
    image = models.ImageField(verbose_name='Image', upload_to='image', null=True, blank=True, default='') ###### 효정이 참고자료
    faceImage = models.ImageField(verbose_name='faceImage', upload_to='image/face', null=True, blank=True, default='')
    petImage = models.ImageField(verbose_name='petImage', upload_to='image/pet', null=True, blank=True, default='')
    foodImage = models.ImageField(verbose_name='foodImage', upload_to='image/food', null=True, blank=True, default='')
    natureImage = models.ImageField(verbose_name='natureImage', upload_to='image/nature', null=True, blank=True, default='')
    created_date = models.DateTimeField(
            default=timezone.now)
    published_date = models.DateTimeField(
            blank=True, null=True)

    def publish(self):
        self.published_date = timezone.now()
        self.save()

    def __str__(self):
        return self.title


# def get_image_url(self):
#     return '%s%s%s' % (settings.MEDIA_ROOT, "/pictures/", self)


#file upload 부분
class Picture(models.Model):
    """This is a small demo using just two fields. The slug field is really not
    necessary, but makes the code simpler. ImageField depends on PIL or
    pillow (where Pillow is easily installable in a virtualenv. If you have
    problems installing pillow, use a more generic FileField instead.

    """
    # file = models.ImageField(upload_to="pictures")
    file = models.ImageField(upload_to="pictures")
    slug = models.SlugField(max_length=50, blank=True)
    image_url = models.URLField(null=True, blank=True)
    def __str__(self):
        return self.file.name

    @models.permalink
    def get_absolute_url(self):
        return ('upload-new', )

    def save(self, *args, **kwargs):
        self.slug = self.file.name
        super(Picture, self).save(*args, **kwargs)

    def delete(self, *args, **kwargs):
        """delete -- Remove to leave file."""
        self.file.delete(False)
        super(Picture, self).delete(*args, **kwargs)


class Etc(models.Model):
    """This is a small demo using just two fields. The slug field is really not
    necessary, but makes the code simpler. ImageField depends on PIL or
    pillow (where Pillow is easily installable in a virtualenv. If you have
    problems installing pillow, use a more generic FileField instead.

    """
    # file = models.ImageField(upload_to="pictures")
    file = models.ImageField(upload_to="etc" , null=True)
    picture = models.ForeignKey(Picture, null=True)
    image_url = models.URLField(null=True, blank=True)
    def __str__(self):
        return self.file.name

    @models.permalink
    def get_absolute_url(self):
        return ('upload-new', )

    def save(self, *args, **kwargs):
        self.slug = self.file.name
        super(Etc, self).save(*args, **kwargs)

    def delete(self, *args, **kwargs):
        """delete -- Remove to leave file."""
        self.file.delete(False)
        super(Etc, self).delete(*args, **kwargs)

class Face(models.Model):
    """This is a small demo using just two fields. The slug field is really not
    necessary, but makes the code simpler. ImageField depends on PIL or
    pillow (where Pillow is easily installable in a virtualenv. If you have
    problems installing pillow, use a more generic FileField instead.

    """
    # file = models.ImageField(upload_to="pictures")
    file = models.ImageField(upload_to="face", null=True)
    slug = models.SlugField(max_length=50, null=True)
    picture = models.ForeignKey(Picture, null=True)
    def __str__(self):
        return self.file.name

    @models.permalink
    def get_absolute_url(self):
        return ('upload-new', )

    def save(self, *args, **kwargs):
        self.slug = self.file.name
        super(Face, self).save(*args, **kwargs)

    def delete(self, *args, **kwargs):
        """delete -- Remove to leave file."""
        self.file.delete(False)
        super(Face, self).delete(*args, **kwargs)

class Fashion(models.Model):
    """This is a small demo using just two fields. The slug field is really not
    necessary, but makes the code simpler. ImageField depends on PIL or
    pillow (where Pillow is easily installable in a virtualenv. If you have
    problems installing pillow, use a more generic FileField instead.

    """
    # file = models.ImageField(upload_to="pictures")
    file = models.ImageField(upload_to="fashion", null=True)
    slug = models.SlugField(max_length=50, null=True)
    picture = models.ForeignKey(Picture, null=True)
    def __str__(self):
        return self.file.name

    @models.permalink
    def get_absolute_url(self):
        return ('upload-new', )

    def save(self, *args, **kwargs):
        self.slug = self.file.name
        super(Fashion, self).save(*args, **kwargs)

    def delete(self, *args, **kwargs):
        """delete -- Remove to leave file."""
        self.file.delete(False)
        super(Fashion, self).delete(*args, **kwargs)


class Food(models.Model):
    """This is a small demo using just two fields. The slug field is really not
    necessary, but makes the code simpler. ImageField depends on PIL or
    pillow (where Pillow is easily installable in a virtualenv. If you have
    problems installing pillow, use a more generic FileField instead.

    """
    # file = models.ImageField(upload_to="pictures")
    file = models.ImageField(upload_to="food", null=True)
    slug = models.SlugField(max_length=50, null=True)
    picture = models.ForeignKey(Picture, null=True)
    def __str__(self):
        return self.file.name

    @models.permalink
    def get_absolute_url(self):
        return ('upload-new', )

    def save(self, *args, **kwargs):
        self.slug = self.file.name
        super(Food, self).save(*args, **kwargs)

    def delete(self, *args, **kwargs):
        """delete -- Remove to leave file."""
        self.file.delete(False)
        super(Food, self).delete(*args, **kwargs)


class Nature(models.Model):
    """This is a small demo using just two fields. The slug field is really not
    necessary, but makes the code simpler. ImageField depends on PIL or
    pillow (where Pillow is easily installable in a virtualenv. If you have
    problems installing pillow, use a more generic FileField instead.

    """
    # file = models.ImageField(upload_to="pictures")
    file = models.ImageField(upload_to="nature", null=True)
    slug = models.SlugField(max_length=50, null=True)
    picture = models.ForeignKey(Picture, null=True)
    def __str__(self):
        return self.file.name

    @models.permalink
    def get_absolute_url(self):
        return ('upload-new', )

    def save(self, *args, **kwargs):
        self.slug = self.file.name
        super(Nature, self).save(*args, **kwargs)

    def delete(self, *args, **kwargs):
        """delete -- Remove to leave file."""
        self.file.delete(False)
        super(Nature, self).delete(*args, **kwargs)

class Pet(models.Model):
    """This is a small demo using just two fields. The slug field is really not
    necessary, but makes the code simpler. ImageField depends on PIL or
    pillow (where Pillow is easily installable in a virtualenv. If you have
    problems installing pillow, use a more generic FileField instead.

    """
    # file = models.ImageField(upload_to="pictures")
    file = models.ImageField(upload_to="pet", null=True)
    slug = models.SlugField(max_length=50, null=True)
    picture = models.ForeignKey(Picture, null=True)
    def __str__(self):
        return self.file.name

    @models.permalink
    def get_absolute_url(self):
        return ('upload-new', )

    def save(self, *args, **kwargs):
        self.slug = self.file.name
        super(Pet, self).save(*args, **kwargs)

    def delete(self, *args, **kwargs):
        """delete -- Remove to leave file."""
        self.file.delete(False)
        super(Pet, self).delete(*args, **kwargs)
