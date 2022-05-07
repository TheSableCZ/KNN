import pickle
import numpy as np
import mediapipe as mp
mp_pose = mp.solutions.pose
import cv2
import tensorflow as tf
import os

def load_dataset(
        dataset_label,
        data_path,
        class_labels,
        save_pickle=False,
        read_pickle=False,
        ext=".jpg"
    ):
    """
    :param dataset_label: Train or Test
    :param data_path: Path to data directory
    :param class_labels: Dictionary of class labels and their 1-hot encoded index
    :param save_pickle: whether or not to save the processed data as a pickled file
    :param read_pickle: whether func should read data from file or process raw image data
    :return: tuple of lists containing data samples and corresponding labels
    """

    # Check to see if dataset should be loaded from file
    if read_pickle:
        try:
            return load_dataset_from_pickle(data_path, dataset_label)
        except:
            print(f"Err: pickled data does not exist. Loading data from images")

    # Load raw imags and extract pose skeleton. Will result in 32 keypoints
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    data = []
    labels = []
    image_data = []
    err_counter = 0
    no_landmarks_counter = 0
    for class_name in class_labels.keys():
        path = f"{data_path}{class_name}/{dataset_label}"
        if not os.path.isdir(f"{data_path}{class_name}"):
          continue

        print(f"Loading {dataset_label} data for {class_name}")
        for filename in os.listdir(path):
            if filename.endswith(ext):
                image = cv2.imread(f"{path}/{filename}")
                if image is None:
                    err_counter += 1
                    continue
                results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                if not results.pose_landmarks:
                    # Ignore any images that don't produce skeletons
                    no_landmarks_counter += 1
                    # continue
                    sample = [(0.0, 0.0)] * 33
                else:
                    # Combine landmarks into datasample
                    sample = []
                    for lm in results.pose_landmarks.landmark:
                        # Create sample which is M x 2 where M is the number of keypoints detected and their
                        # x and y coordinates
                        sample.append((lm.x, lm.y))

                data.append(sample)
                #print(f"append sample: {sample}")
                #print(np.array(data).shape)
                # Create label sample
                label_sample = np.zeros(len(class_labels)) #np.zeros(5)
                label_sample[class_labels[class_name]] = 1
                labels.append(label_sample)

                resized_img = cv2.resize(image, (224, 224))
                image_data.append(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))

                # tf_img = tf.image.decode_jpeg(tf.io.read_file(f"{path}/{filename}"))
                # cv2.imshow('tf', tf_img.numpy())
                # cv2.imshow('cv', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                # cv2.waitKey()

    pose.close()

    print(f"err imread: {err_counter} no landmarks: {no_landmarks_counter}")

    # Check to see if this data should be pickled
    if save_pickle:
        save_dataset_to_pickle(data_path, dataset_label, data, labels)

    return np.array(data), np.array(labels), np.array(image_data)

def load_dataset_from_pickle(data_path, dataset_label):
    filename = f"{data_path}pickled_data/{dataset_label}"
    print(f"Loading {dataset_label} data from pickle")

    data_dict = None
    with open(filename, 'rb') as f:
        data_dict = pickle.load(f)
    data = data_dict['data']
    labels = data_dict['labels']
    print(f"Data loaded: {np.array(data).shape}")
    return np.array(data), np.array(labels)

def save_dataset_to_pickle(data_path, dataset_label, data, labels):
    data_dict = {
        'data': data,
        'labels': labels
    }
    os.makedirs(f"{data_path}pickled_data/", exist_ok=True)
    filename = f"{data_path}pickled_data/{dataset_label}"
    # Clear the contents of the file
    open(filename, 'wb').close()

    with open(filename, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f"saved {filename}")

def load_data(config, data_path, class_labels):
    train_data, train_labels, train_images = load_dataset(
        "Train",
        data_path,
        class_labels,
        save_pickle=config["save_pickle"],
        read_pickle=config["read_pickle"]
    )
    test_data, test_labels, test_images = load_dataset(
        "Test",
        data_path,
        class_labels,
        save_pickle=config["save_pickle"],
        read_pickle=config["read_pickle"]
    )
    val_data, val_labels, val_images = load_dataset(
        "Validation",
        data_path,
        class_labels,
        save_pickle=config["save_pickle"],
        read_pickle=config["read_pickle"]
    )

    # train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).batch(32)
    # test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels)).batch(32)
    # val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels)).batch(32)
    def zip_dataset(data, labels, images):
        poses_ds = tf.data.Dataset.from_tensor_slices(data)
        labels_ds = tf.data.Dataset.from_tensor_slices(labels)
        images_ds = tf.data.Dataset.from_tensor_slices(images)

        dataset = tf.data.Dataset.zip(((poses_ds, images_ds), labels_ds))\
            .shuffle(len(list(poses_ds))).batch(32)
        return dataset

    train_dataset = zip_dataset(train_data, train_labels, train_images)
    test_dataset = zip_dataset(test_data, test_labels, test_images)
    val_dataset = zip_dataset(val_data, val_labels, val_images)

    return train_dataset, test_dataset, val_dataset
