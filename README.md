# ComfyUI-CSM-Nodes

Пользовательские узлы для [ComfyUI](https://github.com/comfyanonymous/ComfyUI), реализующие модель `csm` от [SesameAILabs](https://github.com/SesameAILabs/csm) для генерации речи из текста. Весь необходимый код встроен в проект.

## Возможности
- Узел `CSMTextToSpeech`: Генерирует аудиофайл из текста.
- Параметры: текст, путь к модели, выходной каталог, частота дискретизации, ID спикера.

## Требования
- Установленный ComfyUI.
- Python 3.8+.
- Веса модели `csm` (например, `csm_model.pth`), скачанные из [SesameAILabs/csm](https://github.com/SesameAILabs/csm/releases).

## Установка

1. **Клонируйте репозиторий:**
   ```bash
   git clone https://github.com/<ваш-username>/ComfyUI-CSM-Nodes.git
   cd ComfyUI-CSM-Nodes
