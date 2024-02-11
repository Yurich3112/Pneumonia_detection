import torch
import yaml
from roboflow import Roboflow

# Встановлення залежностей і клонування YOLOv5 необхідно виконати через термінал у PyCharm.
# git clone https://github.com/ultralytics/yolov5
# cd yolov5
# git reset --hard fbe67e465375231474a2ad80a4389efc77ecff99
# pip install -r requirements.txt

# Вивід версії torch і доступності CUDA.
print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))

# Скачування набору даних за допомогою Roboflow.
rf = Roboflow(api_key="YOUR_API_KEY") # вставити сюди код з Roboflow - дивитися readme.txt
project = rf.workspace("YOUR_WORKSPACE").project("YOUR_PROJECT") # вставити сюди код з Roboflow - дивитися readme.txt
dataset = project.version("YOUR_VERSION").download("yolov5") # вставити сюди код з Roboflow - дивитися readme.txt


# Завантаження конфігурації моделі з файлу YAML.
with open(dataset.location + "/data.yaml", 'r') as stream:
    num_classes = str(yaml.safe_load(stream)['nc'])

# Замість IPython magic, використовуємо стандартні операції з файлами.
def writetemplate(filename, content):
    with open(filename, 'w') as f:
        f.write(content.format(**globals()))

# Тренування моделі YOLOv5 (це потрібно виконати в командній строці або терміналі PyCharm).
# python train.py --img 416 --batch 16 --epochs 100 --data {dataset.location}/data.yaml --cfg ./models/yolov5s.yaml --weights '' --name yolov5s_results --cache
