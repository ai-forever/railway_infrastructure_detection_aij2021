AITrain Segmentation
==============================
В этой директории находится код, необходимый для инференса задачи сегментации.

### Описание дирректорий и файлов в них:
src  
├── main.py                - точка входа для обучения модели  
├── segmentation_inference - модуль для инференса обученой сети на тестовых данных  
├── utils.py               - модуль, в котором в том числе находится код формирования сабмита для сегментационной части соревнования  
├── ...                    - иные вспомогательные модули, используемые при обучении и инференсе.

### Описание сабмита:
Файл segmentation_predictions.json имеет следующую, схожую с СОСО форматом, структуру:
```
{  
    "images": [  
        "file_name": ...,        # Имя файла  
        "annotations": [  
            {  
                "counts": ...,   # Маска с предсказаниями в rle формате.  
                "size": ...,     # Размер (ширина, высота) маски.
                'class_id': ...  # id класса, которому маска принадлежит. 

            },
            ...
        ]
    ],
    # Категории, с именами классов и их id.
    "categories": [
        {"supercategory": "railway_object","id": 0,"name": "MainRailPolygon"},
        {"supercategory": "railway_object","id": 1,"name": "AlternativeRailPolygon"},
        {"supercategory": "railway_object","id": 2,"name": "Train"},
    ]
    }
```

### HOW TO:
#### Запустить тренировку сегментационного бейзлайна
```bash
# В окружении с установленными библиотеками из requirements.txt
# из корня репозитория запустить, обучение производилось на одной 1080TI
PYTHONPATH=. python segmentation/src/main.py \
                    --path_to_data path/to/data/folder \
                    --logdir path/where/to/store/logs

```