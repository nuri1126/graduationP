# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('blog', '0007_auto_20170322_1450'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='picture',
            name='published_date',
        ),
        migrations.AlterField(
            model_name='picture',
            name='file',
            field=models.ImageField(default=1, upload_to='pictures'),
            preserve_default=False,
        ),
    ]
