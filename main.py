# Kishen Das
# Metric Based Image Aesthetics Assessment
# Trained on AADB dataset, with derived photographics feature and metrics
# OpenCV used for image processing
#function to import libraries
#import
import cv2
import numpy as np
import torch






#Evaluate image Colors
def evaluate_colors(img):
    #Analyse colors
    #get color histogram
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    #normalize histogram
    hist = cv2.normalize(hist, hist).flatten()
    #get color contrast
    contrast = cv2.Laplacian(img, cv2.CV_64F).var()
    #get color entropy
    entropy = cv2.calcHist([img], [0], None, [256], [0, 256])
    entropy = -np.sum([p * np.log(p) for p in entropy if p != 0])
    #get color saturation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = np.mean(hsv[:, :, 1])
    #get color brightness
    brightness = np.mean(hsv[:, :, 2])
    #get color hue
    hue = np.mean(hsv[:, :, 0])
    #get color sharpness
    sharpness = cv2.Laplacian(img, cv2.CV_64F).var()
    #get color homogeneity
    homogeneity = cv2.Laplacian(img, cv2.CV_64F).var()
    #get color energy
    energy = cv2.Laplacian(img, cv2.CV_64F).var()
    #get color correlation
    correlation = cv2.Laplacian(img, cv2.CV_64F).var()
    #get color variance
    variance = cv2.Laplacian(img, cv2.CV_64F).var()
    #get color mean
    mean = cv2.Laplacian(img, cv2.CV_64F).var()
    #return color features
    return hist, contrast, entropy, saturation, brightness, hue, sharpness, homogeneity, energy, correlation, variance, mean

#Evaluate image focus
def evaluate_focus_crop(img):
    def sharpen_image(image):
        sharp_img = cv2.filter2D(image, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
        return sharp_img

    def canny_segmentation(img, low_threshold=100, high_threshold=200):
        edges = cv2.Canny(img, low_threshold, high_threshold)
        return edges

    def get_bounding_box(image, thresh=0.95):
        nonzero_indices = np.nonzero(image.T)
        min_row, max_row = np.min(nonzero_indices[0]), np.max(nonzero_indices[0])
        min_col, max_col = np.min(nonzero_indices[1]), np.max(nonzero_indices[1])
        box_size = max_row - min_row + 1, max_col - min_col + 1
        box_size_thresh = (int(box_size[0] * thresh), int(box_size[1] * thresh))
        # box_size_thresh = (int(box_size[0]), int(box_size[1]))
        # coordinates of the box that contains 95% of the highest pixel values
        top_left = (
        min_row + int((box_size[0] - box_size_thresh[0]) / 2), min_col + int((box_size[1] - box_size_thresh[1]) / 2))
        bottom_right = (top_left[0] + box_size_thresh[0], top_left[1] + box_size_thresh[1])
        return (top_left[0], top_left[1]), (bottom_right[0], bottom_right[1])

    def is_blurry(image, thresh=150, crop_edges_thresh=0.75, canny_thresh_low=100, canny_thresh_high=200):
        if (len(image.shape) < 3):
            gray = image
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        gray = sharpen_image(gray)
        seg = canny_segmentation(gray, canny_thresh_low, canny_thresh_high)
        bb_thresh = get_bounding_box(seg, crop_edges_thresh)
        im_crop = gray[bb_thresh[0][1]:bb_thresh[1][1], bb_thresh[0][0]:bb_thresh[1][0]]
        edges = cv2.Laplacian(im_crop, cv2.CV_64F)
        return edges.var() < thresh

#Function to segment image via panoptic segmentation detr model, returns mask, bounding box, and class
def segment_image(img):
    #load model
    model = torch.hub.load('facebookresearch/detr', 'detr_resnet101_panoptic', pretrained=True)
    #load image
    img = Image.open(img)
    #transform image
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[1, 1, 1]),
    ])
    #transform image
    img = transform(img)
    #put image into batch
    img = img.unsqueeze(0)
    #load model
    model.eval()
    #get prediction
    outputs = model(img)
    #get segmentation
    segmentations = outputs['pred_segments']
    #get bounding box
    boxes = outputs['pred_boxes']
    #get class
    classes = outputs['pred_classes']
    #return segmentation, bounding box, and class
    return segmentations, boxes, classes

#function iteratively create crops based on bounding boxes and evaluate focus and color returning color and focus data
def crop_image(img, boxes):
    #create empty lists
    focus = []
    color = []
    #iterate through bounding boxes
    for i in range(len(boxes[0])):
        #get bounding box
        box = boxes[0][i]
        #get crop
        crop = img[box[1]:box[3], box[0]:box[2]]
        #evaluate focus
        focus.append(evaluate_focus_crop(crop))
        #evaluate color
        color.append(evaluate_colors(crop))
    #return focus and color data
    return focus, color
#fuction to resize image for direct CCN input
def resize_image(img):
    #resize image
    img = cv2.resize(img, (224, 224))
    #return image
    return img

#function to preprocess images with segmentation cropping and resizing
def preprocess_image(img):
    #load image
    img = cv2.imread(img)
    #segment image
    segmentations, boxes, classes = segment_image(img)
    #crop image
    focus, color = crop_image(img, boxes)
    #resize image
    img = resize_image(img)
    #return image, focus, and color data
    return img, focus, color, segmentations, classes


#function to create input vector for CNN from img, focus, color, and segmentation data
def create_input_vector(img, focus, color, segmentations, classes):
    #create empty list
    input_vector = []
    #append image to input vector
    input_vector.append(img)
    #append focus data to input vector
    input_vector.append(focus)
    #append color data to input vector
    input_vector.append(color)
    #append segmentation data to input vector
    input_vector.append(segmentations)
    #append class data to input vector
    input_vector.append(classes)
    #normalize input vector
    input_vector = tf.keras.utils.normalize(input_vector, axis=1)
    #return input vector
    return input_vector

#function to preprocess images on csv list
def preprocess_dataset(dataset):
    #create empty list
    input_vector = []
    #iterate through dataset
    for i in range(len(dataset)):
        #get image
        img = dataset['image'][i]
        #preprocess image
        img, focus, color, segmentations, classes = preprocess_image(img)
        #create input vector
        input_vector.append(create_input_vector(img, focus, color, segmentations, classes))
    #return input vector
    return input_vector

#define training model for CNN with a linear output layer input shapes are 512,512,4 and a vector of length 100
def define_model():
    #create model
    model = tf.keras.models.Sequential()
    #add convolutional layer
    model.add(tf.keras.layers.Conv2D(64, (3, 3), input_shape=(512, 512, 4)))
    #add activation function
    model.add(tf.keras.layers.Activation('relu'))
    #add pooling layer
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    #add convolutional layer
    model.add(tf.keras.layers.Conv2D(64, (3, 3)))
    #add activation function
    model.add(tf.keras.layers.Activation('relu'))
    #add pooling layer
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    #add flattening layer
    model.add(tf.keras.layers.Flatten())
    #add dense layer
    model.add(tf.keras.layers.Dense(64))
    #add activation function
    model.add(tf.keras.layers.Activation('relu'))
    #add dropout layer
    model.add(tf.keras.layers.Dropout(0.5))
    #add dense layer
    model.add(tf.keras.layers.Dense(1))
    #add activation function
    model.add(tf.keras.layers.Activation('linear'))
    #compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #return model
    return model

#train model
def train_model(model, input_vector, dataset):
    #create empty list
    y = []
    #iterate through dataset
    for i in range(len(dataset)):
        #get label
        label = dataset['label'][i]
        #append label to list
        y.append(label)
    #fit model
    model.fit(input_vector, y, epochs=10)
    #return model
    return model
#test model
def test_model(model, input_vector, dataset):
    #create empty list
    y = []
    #iterate through dataset
    for i in range(len(dataset)):
        #get label
        label = dataset['label'][i]
        #append label to list
        y.append(label)
    #evaluate model
    val_loss, val_acc = model.evaluate(input_vector, y)
    #print accuracy
    print(val_acc)
    #return accuracy
    return val_acc

#run model
train_model(create_model(), preprocess_dataset(dataset), dataset[2])

#test model
test_model(create_model(), preprocess_dataset(dataset), dataset[2])






