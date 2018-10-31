# -*- coding: utf-8 -*-
# Generated by Django 1.10.5 on 2017-02-20 04:25
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('blog', '0004_post_image'),
    ]

    operations = [
        migrations.AddField(
            model_name='post',
            name='faceImage',
            field=models.ImageField(blank=True, default='', null=True, upload_to='image/face', verbose_name='faceImage'),
        ),
        migrations.AddField(
            model_name='post',
            name='foodImage',
            field=models.ImageField(blank=True, default='', null=True, upload_to='image/food', verbose_name='foodImage'),
        ),
        migrations.AddField(
            model_name='post',
            name='natureImage',
            field=models.ImageField(blank=True, default='', null=True, upload_to='image/nature', verbose_name='natureImage'),
        ),
        migrations.AddField(
            model_name='post',
            name='petImage',
            field=models.ImageField(blank=True, default='', null=True, upload_to='image/pet', verbose_name='petImage'),
        ),
        migrations.AlterField(
            model_name='post',
            name='image',
            field=models.ImageField(blank=True, default='', null=True, upload_to='image', verbose_name='Image'),
        ),
    ]
