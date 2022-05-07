from utils.dataloader import load_data
from utils.model import create_model, predict_with_static_image, predict_with_video, train_model, \
    create_dense_pose_model
from tensorflow import keras

"""class_labels = {
    "Warrior_I": 0,
    "Warrior_II": 1,
    "Tree": 2,
    "Triangle": 3,
    "Standing_Splits": 4
}"""

class_labels = {'Chair_Pose_or_Utkatasana_': 0,
                'Child_Pose_or_Balasana_': 1,
                'Cobra_Pose_or_Bhujangasana_': 2,
                'Downward-Facing_Dog_pose_or_Adho_Mukha_Svanasana_': 3,
                'Four-Limbed_Staff_Pose_or_Chaturanga_Dandasana_': 4,
                'Happy_Baby_Pose_or_Ananda_Balasana_': 5,
                'Intense_Side_Stretch_Pose_or_Parsvottanasana_': 6,
                'Low_Lunge_pose_or_Anjaneyasana_': 7,
                'Plank_Pose_or_Kumbhakasana_': 8,
                'Standing_Forward_Bend_pose_or_Uttanasana_': 9,
                'Warrior_I_Pose_or_Virabhadrasana_I_': 10}

data_path = "../yoga11/"

config = {
    "create_model": True,
    "load_model": False,
    "train_model": True,
    "eval_model": True,
    "predict_static": True,
    "predict_video": False,
    "read_pickle": True,
    "save_pickle": False,
    "display_stats": False
}

if __name__ == "__main__":
    train_dataset, test_dataset, val_dataset = load_data(config, data_path, class_labels)

    import matplotlib.pyplot as plt
    import numpy as np
    for (pose, images), labels in train_dataset.take(3):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(list(class_labels.keys())[np.argmax(labels[i])][:8])
            plt.axis("off")
        plt.show()

    model = None
    if config["create_model"]:
        model = create_dense_pose_model(len(class_labels))  # create_model()
    if config["load_model"]:
        model = keras.models.load_model("path/to/saved/models")
    if config["train_model"]:
        train_model(model, config, train_dataset, val_dataset)
    if config["eval_model"]:
        loss, acc = model.evaluate(test_dataset)
        print(f"loss: {loss}\nacc: {acc}")
    if config["predict_static"]:
        predict_with_static_image(model, class_labels, data_path)
    if config["predict_video"]:
        predict_with_video(model, class_labels)
