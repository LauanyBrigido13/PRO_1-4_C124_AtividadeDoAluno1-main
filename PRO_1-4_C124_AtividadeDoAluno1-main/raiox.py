import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img, ImageDataGenerator
import matplotlib.pyplot as plt

#Carregue as imagens
image = img_to_array(load_img("https://vscode.dev/github/LauanyBrigido13/PRO_1-4_C124_AtividadeDoAluno1-main/blob/main/PRO_1-4_C124_AtividadeDoAluno1-main/PRO_1-1_C111_PneumothoraxImageDataset/training_dataset/infected/image_1.png",target_size=(180,180)))

#Crie um gerador de imagem
data_generator = ImageDataGenerator(rotation_range = 90, fill_mode = 'nearest')

#Expandir a imagem
image =tf.expand_dims(image, axis=0)

#Gere imagens aumentadas
aumented_image = data_generator.flow(image)

#Visualize a imagem aumentada
plt.imshow(array_to_img(aumented_image[0][0]))
plt.show()
