'''
    Questo file.py contiene tutte le classi e le funzioni necessarie per la
    creazione, addestramento e salvataggio della rete neurale.
'''
# imports
from tensorflow import keras
from keras import models, layers
from src import digit_recognition

# funzione per creare il modello
def create_model(num_classes):
    '''
        Funzione che istanzia un modello di rete neurale attraverso Keras.
    '''
    model = models.Sequential()

    model.add(layers.Input(shape=784,))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax')) # output layer

    return model

# valutazione del modello
def grid_evaluation(model, grid_path, grid_labels):
  digits, digits_images = extract_and_preprocess(grid_path)
  classifieds = []
  print(grid_labels)
  classified = 0
  for i in range(len(digits)):    
    predict_digit = model.predict(digits[i:i+1])
    class_digit = np.argmax(predict_digit,axis=1)
    
    if class_digit == grid_labels[i]:
      classified += 1
    #else:
      #if grid_labels[i] == 0:
        #print("sono qui")
        #predict_digit = model_s.predict(digits[i:i+1])
        #class_digit = np.argmax(predict_digit,axis=1)
        #if class_digit == grid_labels[i]:
        #  classified += 1
    plt.imshow(cv2.cvtColor(digits_images[i], cv2.COLOR_RGB2BGR))
    plt.show()
    print(class_digit)
      
      #print("class_digit: {} | predict_digit: {}".format(class_digit, predict_digit))
    classifieds.append(class_digit)
  print(classifieds)
  return classified / len(grid_labels)


# funzione per addestrare il modello e salvarlo


# funzione per salvarlo (necessario?)


# funzione per addestrare il modello


#