# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('blog', '0006_picture'),
    ]

    operations = [
        migrations.AddField(
            model_name='picture',
            name='published_date',
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='picture',
            name='file',
            field=models.ImageField(blank=True, upload_to='image', default='', verbose_name='Image', null=True),
        ),
    ]
