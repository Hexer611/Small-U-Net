import cv2.cv2 as cv
from tensorflow import keras
import numpy as np
import os
import imutils


def my_save_model(model, model_folder, model_name):
    """
        Saves given model to given folder with given name. Saves can stack up to 5 files. Ex: save1.h5,save2.h5,...
    """
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
    times = []
    number_of_files = 0
    for r, d, f in os.walk(model_folder):
        for file in f:
            if str(file).__contains__(model_name):
                times.append(os.path.getmtime(os.path.join(r, file)))
                number_of_files += 1
    if number_of_files < 5:
        model.save(model_folder + model_name + str(number_of_files + 1) + '.h5')
        print("Successfully saved to here => " + model_folder + model_name + str(number_of_files + 1) + '.h5')
    else:
        model.save(model_folder + model_name + str(times.index(min(times)) + 1) + '.h5')
        print("Successfully saved to here => " + model_folder + model_name + str(times.index(min(times)) + 1) + '.h5')


def my_load_model(model_folder, model_name):
    """
        Loads latest save among five or less files from given folder with respect to given model name.
    """
    from my_model_init import my_IoU
    times = []
    for r, d, f in os.walk(model_folder):
        print(f)
        for file in f:
            if str(file).__contains__(model_name):
                times.append(os.path.getmtime(os.path.join(r, file)))
    load = keras.models.load_model(model_folder + model_name + str(times.index(max(times)) + 1) + '.h5',
                                   custom_objects={'my_IoU': my_IoU})
    print("Successfully loaded this save => " + model_folder + model_name + str(times.index(max(times)) + 1) + '.h5')
    return load


def my_train_model(model, save_model_folder, _input, _output, model_name, saving=False, tensorboard_file=None, val=False,
                   batch_size=32, epochs=20, validation_data=None):
    """
        Training happens here along with setting parameters.
    """
    if tensorboard_file is not None:
        log_dir = tensorboard_file
        tensor_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=50, profile_batch=100000)
        if validation_data is not None:
            model.fit(_input, _output, epochs=epochs, callbacks=[tensor_callback], batch_size=batch_size, validation_data=validation_data)
        else:
            model.fit(_input, _output, epochs=epochs, callbacks=[tensor_callback], batch_size=batch_size)
    else:
        if validation_data is not None:
            model.fit(_input, _output, epochs=epochs, batch_size=batch_size, validation_data=validation_data)
        else:
            model.fit(_input, _output, epochs=epochs, batch_size=batch_size)
    if saving:
        my_save_model(model, save_model_folder, model_name)
    return model


def my_fullscreen_semantic_prediction(original_image, model, desired_image_size_x=1280, desired_image_size_y=720, screen_division=False, classes=10):
    """
        If screen_division = True
            Divides screen into pieces and does prediction for each of them and returns the whole prediction
        Else
            Does mask prediction based on given image
    """
    if original_image.shape[2] > 3:
        original_image = original_image[:, :, 0:3]
    blocks = []
    if screen_division:
        for i in range(2):
            for j in range(4):
                blocks.append(cv.resize(original_image[i*540:(i+1)*540, j*480:(j+1)*480, :], (448, 448)))
        blocks = np.array(blocks)
        blocks = blocks / 256

        _output_0 = model.predict(blocks)

        output_mask = cv.vconcat([cv.hconcat(_output_0[0:4]), cv.hconcat(_output_0[4:8])])
        output_mask = cv.resize(output_mask, (desired_image_size_x, desired_image_size_y), interpolation=cv.INTER_NEAREST)

        output = np.zeros((720, desired_image_size_x, 3), dtype=np.uint8)

        for i in range(classes):
            output[..., 1] = np.where(output_mask[..., i] > 0.75, 255, output[..., 1])

        original_image = cv.resize(original_image, (desired_image_size_x, desired_image_size_y), interpolation=cv.INTER_NEAREST)
    else:
        input = cv.resize(original_image, (448, 448))
        _output_0 = model.predict(input[np.newaxis, ...] / 256)
        output = np.zeros((448, 448, 3), dtype=np.uint8)
        colors = np.array([[255,255,1],[1,255,1],[1,1,255],[255,128,255],[255,81,168],[255,63,98],[183,255,112],[82,255,245],[255,86,86],[112,183,255]])
        for _ in range(classes):
            output[_output_0[0, ..., _] >= 0.75] = colors[_]
        output = cv.resize(output, (desired_image_size_x, desired_image_size_y), interpolation=cv.INTER_NEAREST)
        original_image = cv.resize(original_image, (desired_image_size_x, desired_image_size_y), interpolation=cv.INTER_CUBIC)
    mask = cv.inRange(output, (0, 0, 0), (0, 0, 0))
    ret, mask = cv.threshold(mask, 0, 255, cv.THRESH_BINARY)
    inv_mask = cv.bitwise_not(mask)

    resized_image = cv.bitwise_and(original_image, output, mask=inv_mask)
    return resized_image


def my_semantic_prediction(image, model=None, threshold=0.5):
    """
        Takes input image along with model and does the prediction and returns it.
    """
    p_image = image.copy()
    p_image = cv.resize(p_image, (448, 448))
    p_image = p_image / 255.0

    image_mask = model.predict(np.reshape(p_image, (1, 448, 448, 3)).astype('float16'))
    image_mask = np.reshape(image_mask[..., 0], (448, 448, 1))

    image_mask[image_mask >= threshold] = 255
    image_mask[image_mask < (1 - threshold)] = 0

    image_mask = cv.cvtColor(image_mask, cv.COLOR_GRAY2BGR)
    p_image = p_image * 255
    p_image = p_image.astype(np.uint8)
    return image_mask, p_image


def pixel_mask_image_generator(foreground_input_folder, background_input_folder, output_folder, create_image_count=1024,
                               image_name='image', fore_ground_count=1, background_image_count=5063, input_size=448, output_size=448):
    """
        Dataset generator using data augmentation.
        Takes object picures and places them in random backgrounds.
    """
    import random as rn
    rn.seed(None)
    background_input_folder = background_input_folder + image_name
    foreground_input_folder = foreground_input_folder + image_name
    output_folder = output_folder + image_name

    for x in range(create_image_count):
        print(x)
        writing_path = output_folder + str(x).rjust(6, '0') + '.png'
        mask_writing_path = output_folder + str(x).rjust(6, '0') + '_mask' + '.png'
        class_writing_path = output_folder + str(x).rjust(6, '0') + '_class' + '.txt'

        read_index_background = rn.randint(1, background_image_count)
        background_reading_path = background_input_folder + str(read_index_background).rjust(5, '0') + '.jpg'

        background = cv.imread(background_reading_path, cv.IMREAD_UNCHANGED)
        background = cv.cvtColor(background, cv.COLOR_BGR2BGRA)
        background = cv.resize(background, (input_size, input_size), interpolation=cv.INTER_NEAREST)

        read_index_foreground = rn.randint(1, fore_ground_count)
        foreground_reading_path = foreground_input_folder + str(read_index_foreground).rjust(3, '0') + '.png'
        image = cv.imread(foreground_reading_path, cv.IMREAD_UNCHANGED)
        scale_x = 0.9 + rn.random() * 0.6
        scale_y = scale_x * (0.9 + rn.random() * 0.2)
        image = cv.resize(image, (int(scale_y*50), int(scale_x*80)), interpolation=cv.INTER_NEAREST)
        rotation_angle = 360 * rn.random()
        image = imutils.rotate_bound(image, rotation_angle)

        background_mask = np.zeros((background.shape[0],background.shape[1], 3), np.uint8)

        offset_x = rn.randint(20, input_size-58)
        offset_y = rn.randint(20, input_size-58)

        for _X in range(image.shape[0]):
            for _Y in range(image.shape[1]):
                is_inside_background = 0 <= offset_x + _X < background.shape[0] and 0 <= offset_y + _Y < background.shape[1]
                if image[_X, _Y, 3] != 0 and is_inside_background:
                    # Adds object to the background according to the alpha channel.
                    background[offset_x + _X, offset_y + _Y, 0:3] = image[_X, _Y, 0:3]
                    background_mask[offset_x + _X, offset_y + _Y, 0:3] = 255

        background_mask = cv.resize(background_mask, (output_size, output_size))

        cv.imwrite(writing_path, background)
        cv.imwrite(mask_writing_path, background_mask)
        f = open(class_writing_path, "w")
        f.write(str(read_index_foreground-1))
        f.close()


def my_tiff_input_output_init(train_inp_fol, train_out_fol,  val_inp_fol=None, val_out_fol=None, image_dimensions=448, output_shape=448, classes=1, image_count=10, file_extension='.tif', validation=False):
    """
        Can initialize input images for both training and validation.
        Input images can have any size but must be rgb images with the file extension in the parameters.
        This method is designed for two class cases where one class is the object to be predicted and the other is background.
        Output mask images must have a 0 pixel value if it is background, non-zero if it will be detected.
        Files must have names like 000.tif - 001.tif etc.
    """
    _input = np.zeros((image_count, image_dimensions, image_dimensions, 3), dtype=np.float32)
    _output = np.zeros((image_count, output_shape, output_shape, classes + 1), dtype=np.float32)
    for image_number in range(image_count):
        print(image_number)
        print(train_inp_fol + str(image_number).rjust(3, '0') + file_extension)
        t_image = cv.imread(train_inp_fol + str(image_number).rjust(3, '0') + file_extension, -1)
        t_image = np.uint8(t_image)
        t_image = cv.cvtColor(t_image, cv.COLOR_GRAY2BGR)
        print(train_out_fol + str(image_number).rjust(3, '0') + file_extension)
        mask_image = cv.imread(train_out_fol + str(image_number).rjust(3, '0') + file_extension, -1)
        mask_image[mask_image != 0] = 1
        mask_image = np.uint8(mask_image)
        t_image = cv.resize(t_image, (image_dimensions, image_dimensions))
        mask_image = cv.resize(mask_image, (output_shape, output_shape))

        _input[image_number] = t_image / 255.0
        _output[image_number, ..., 0] = mask_image
        _output[image_number, ..., 1] = 1 - mask_image
    if validation:
        val_input = np.zeros((image_count, image_dimensions, image_dimensions, 3), dtype=np.float32)
        val_output = np.zeros((image_count, output_shape, output_shape, classes + 1), dtype=np.float32)
        for image_number in range(image_count):
            print(image_number)
            print(val_inp_fol + str(image_number).rjust(3, '0') + file_extension)
            t_image = cv.imread(val_inp_fol + str(image_number).rjust(3, '0') + file_extension, -1)
            t_image = np.uint8(t_image)
            t_image = cv.cvtColor(t_image, cv.COLOR_GRAY2BGR)
            mask_image = cv.imread(val_out_fol + str(image_number).rjust(3, '0') + file_extension, -1)
            mask_image[mask_image != 0] = 1
            mask_image = np.uint8(mask_image)
            t_image = cv.resize(t_image, (image_dimensions, image_dimensions))
            mask_image = cv.resize(mask_image, (output_shape, output_shape))

            val_input[image_number] = t_image / 255.0
            val_output[image_number, ..., 0] = mask_image
            val_output[image_number, ..., 1] = 1 - mask_image
        return _input, _output, val_input, val_output
    return _input, _output


def my_grayscale_input_output_init(train_inp_fol, train_out_fol, image_dimensions=448, output_shape=448, classes=34, image_count=200, file_extension='.png'):
    """
        Can initialize input images for training.
        Input images can have any size but must be rgb images with the file extension in the parameters.
        In output images pixel values define class number of that pixel.
        Output images must be grayscale.
        Files must have names like 000000.tif - 000001.tif etc.
    """
    _input = np.zeros((image_count, image_dimensions, image_dimensions, 3), dtype=np.float32)
    _output = np.zeros((image_count, output_shape, output_shape, classes), dtype=np.float32)
    for image_number in range(image_count):
        rgb_image_path = train_inp_fol + str(image_number).rjust(6, '0') + file_extension
        mask_image_path = train_out_fol + str(image_number).rjust(6, '0') + file_extension

        t_image = cv.imread(rgb_image_path, -1)
        mask_image = cv.imread(mask_image_path, cv.IMREAD_GRAYSCALE)
        t_image = cv.resize(t_image, (image_dimensions, image_dimensions), cv.INTER_NEAREST)
        mask_image = cv.resize(mask_image, (output_shape, output_shape), cv.INTER_NEAREST)
        for _ in range(classes):
            _output[image_number, ..., _] = np.where(mask_image == _, 1, 0)
        _input[image_number] = t_image / 255.0
    return _input, _output
