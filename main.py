import numpy as np
import cv2.cv2 as cv
import my_model_init
import helper_functions as hp
import mss
from tensorflow import config

# GPU memory error solve thing
gpus = config.experimental.list_physical_devices('GPU')
config.experimental.set_memory_growth(gpus[0], True)
# ----------------------------------------------------------

# Folder parameters
model_folder = 'saved_models/'
tensorboard_log_folder = 'tensorboard_logs/'
image_folder = 'PhC-C2DH-U373/01/t' # 'DIC-C2DH-HeLa/01/t'
train_inp_fol = 'PhC-C2DH-U373/01/t' # 'DIC-C2DH-HeLa/01/t'
train_out_fol = 'PhC-C2DH-U373/01_GT/TRA/man_track' # 'DIC-C2DH-HeLa/01_GT/TRA/man_track'
# ----------------------------------------------------------


def my_main_jet_net():
    # ----------------parameters-----------------------
    train = False
    load = True
    iterate_over_dataset = True
    screen_capture = False
    input_shape = 448
    output_shape = 448
    model_name = 'phc-u373-new' # 'dic-hela-new'

    # Wether load the model or create new one
    if load:
        model = hp.my_load_model(model_folder, model_name)
        my_model_init.my_recompiler(model, lr=0.0001)
    else:
        model = my_model_init.my_small_net(input_shape, 3, output_shape, classes=1)

    # Training
    if train:
        _inp, _output = hp.my_tiff_input_output_init(train_inp_fol, train_out_fol, image_dimensions=input_shape, output_shape=output_shape,image_count=115, classes=1)
        for x in range(1):
            print('Train index  : ' + str(x))
            model = hp.my_train_model(model, model_folder, _inp, _output, val=False,
                                      saving=True,
                                      batch_size=2,
                                      model_name=model_name,
                                      epochs=200, tensorboard_file=tensorboard_log_folder + '/' + model_name + '/',
                                      # validation_data=(val_inp, val_output)
                                      )

    if iterate_over_dataset:
        test_image_count = 84
        for x in range(test_image_count):
            image_path = image_folder + str(x).rjust(3, '0') + '.tif'
            print(image_path)

            original_image = cv.imread(image_path, -1)
            original_image = cv.cvtColor(original_image, cv.COLOR_GRAY2BGR)
            image_mask, p_image = hp.my_semantic_prediction(original_image, model)
            cv.imwrite('/predict/mask' + str(x).rjust(3, '0') + '.png', image_mask)
            cv.imshow('mask', image_mask)
            cv.imshow('real', p_image)
            cv.waitKey(1)

    if screen_capture:
        sct = mss.mss()
        mon2 = sct.monitors[1]
        print(mon2['top'])
        print(mon2['left'])
        mon = {'top': mon2['top'], 'left': mon2['left'], 'width': 1920, 'height': 1080}
        while True:
            screen = np.array(sct.grab(mon))
            #screen = np.flip(screen[:, :, :3], 2)
            #Image.frombytes('BGR', screen.width, screen.height, 'raw', 'BGRX')
            screen = hp.my_fullscreen_semantic_prediction(screen, model, screen_division=False, classes=1)
            cv.imshow('konnichiwa', screen)
            if cv.waitKey(1) & 0xFF == ord('a'):
                break


new_image_generate = False
if new_image_generate:
    image_count = 2048
    foreground_input_folder = "/example/object/folder/"
    background_input_folder = "/example/background/folder/"
    output_folder = "/example/output/folder/"

    hp.pixel_mask_image_generator(foreground_input_folder, background_input_folder, output_folder, image_count, fore_ground_count=10, input_size=448, output_size=448)

my_main_jet_net()
