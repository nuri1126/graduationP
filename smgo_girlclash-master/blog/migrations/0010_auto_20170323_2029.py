# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('blog', '0009_etc_faces_fashion_food_nature_pets'),
    ]

    operations = [
        migrations.RenameModel(
            old_name='Faces',
            new_name='Face',
        ),
        migrations.RenameModel(
            old_name='Pets',
            new_name='Pet',
        ),
    ]
