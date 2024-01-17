from keras.preprocessing.image import ImageDataGenerator
from diag import summarize_diagnostics
from utils import *
import time

def run_test_harness(model, tensorboard_callback, augmentation):
    if augmentation:
        train_datagen = ImageDataGenerator(rescale=1.0/255.0,
        width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
        test_datagen = ImageDataGenerator(rescale=1.0/255.0)

        train_it = train_datagen.flow_from_directory('dataset/train/',
            class_mode='binary', batch_size=40, target_size=(200, 200))
        test_it = test_datagen.flow_from_directory('dataset/test/',
            class_mode='binary', batch_size=40, target_size=(200, 200))
    else:
        datagen = ImageDataGenerator(rescale=1.0/255.0)
        train_it = datagen.flow_from_directory('dataset/train/',
        class_mode='binary', batch_size=40, target_size=(200, 200))
        test_it = datagen.flow_from_directory('dataset/test/',
        class_mode='binary', batch_size=40, target_size=(200, 200))
    start = time.time()
    history = model.fit(train_it, steps_per_epoch=len(train_it),
    validation_data=test_it, validation_steps=len(test_it), epochs=6, callbacks = [tensorboard_callback], verbose=0)
    end = time.time()
    print("Time taken for Training: ", end-start)
    train_loss, train_acc = model.evaluate(train_it, steps=len(train_it), verbose=0)
    _, acc = model.evaluate(test_it, steps=len(test_it), verbose=0)
    print('Training loss: %.3f, Training accuracy: %.3f' % (train_loss, train_acc*100.0))
    print('Testing accuracy: %.3f' % (acc*100.0))
    print('Model Parameters: ', model.count_params())
    #  print('> %.3f' % (acc * 100.0))

    writer = tf.summary.create_file_writer("logs_mlp/validation/")
    step = 0
    preds = model.predict(test_it)
    preds = [0 if preds[i]<0.5 else 1 for i in range(len(preds))]
    class_names = ['Dolphin', 'Jellyfish']

    for batch_idx, (x, y) in enumerate(test_it):
        figure = image_grid(x, preds, class_names)
        print(batch_idx)
        with writer.as_default():
            tf.summary.image(
                "Visualize Images", plot_to_image(figure), step=step,
            )
            step += 1
        if(batch_idx == 0):
            break
        summarize_diagnostics(history)