# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('blog', '0008_auto_20170322_1728'),
    ]

    operations = [
        migrations.CreateModel(
            name='Etc',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, primary_key=True, auto_created=True)),
                ('file', models.ImageField(upload_to='pictures')),
                ('slug', models.SlugField(blank=True)),
                ('picture', models.ForeignKey(to='blog.Picture')),
            ],
        ),
        migrations.CreateModel(
            name='Faces',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, primary_key=True, auto_created=True)),
                ('file', models.ImageField(upload_to='pictures')),
                ('slug', models.SlugField(blank=True)),
                ('picture', models.ForeignKey(to='blog.Picture')),
            ],
        ),
        migrations.CreateModel(
            name='Fashion',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, primary_key=True, auto_created=True)),
                ('file', models.ImageField(upload_to='pictures')),
                ('slug', models.SlugField(blank=True)),
                ('picture', models.ForeignKey(to='blog.Picture')),
            ],
        ),
        migrations.CreateModel(
            name='Food',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, primary_key=True, auto_created=True)),
                ('file', models.ImageField(upload_to='pictures')),
                ('slug', models.SlugField(blank=True)),
                ('picture', models.ForeignKey(to='blog.Picture')),
            ],
        ),
        migrations.CreateModel(
            name='Nature',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, primary_key=True, auto_created=True)),
                ('file', models.ImageField(upload_to='pictures')),
                ('slug', models.SlugField(blank=True)),
                ('picture', models.ForeignKey(to='blog.Picture')),
            ],
        ),
        migrations.CreateModel(
            name='Pets',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, primary_key=True, auto_created=True)),
                ('file', models.ImageField(upload_to='pictures')),
                ('slug', models.SlugField(blank=True)),
                ('picture', models.ForeignKey(to='blog.Picture')),
            ],
        ),
    ]
