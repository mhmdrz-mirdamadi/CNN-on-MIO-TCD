# Deep CNN with PyTorch on mini MIO-TCD dataset

The [MIO-TCD](https://tcd.miovision.com/challenge/dataset.html) dataset consists of total 786,702 images with 648,959 in the classification dataset and 137,743 in the localization dataset acquired at different times of the day and different periods of the year by thousands of traffic cameras deployed all over Canada and the United States.

This dataset is a mini version of MIO-TCD, consisting of 25000 images in 5 classes, and could be downloaded [here](https://drive.google.com/file/d/1RrpJ76xVtVgTD4cnFrzA7bMrwM-9_xfZ/view?usp=drive_link).

|      **Class**      | **Train** | **Valid** | **Test** |
| :-----------------: | :-------: | :-------: | :------: |
| _articulated_truck_ |   3000    |   1000    |   1000   |
|    _background_     |   3000    |   1000    |   1000   |
|        _bus_        |   3000    |   1000    |   1000   |
|        _car_        |   3000    |   1000    |   1000   |
|     _work-van_      |   3000    |   1000    |   1000   |
|      **Total**      | **15000** | **5000**  | **5000** |

This CNN was implemented in `PyTorch` with logging and hyperparameter tuning in `W&B` and consists of:

1. An underfit model
2. An Overfit model
3. Best fit model (Grid search for `learning_rate` with `Skorch` and `W&B`)
4. Transfer learning with `ResNet18` (Freezing weights)
5. Transfer learning with `ResNet18` (Fine tune the full CNN)
6. Evaluation metrics

## Evaluation metrics

|                 | **Train** | **Valid** | **Test** |
| :-------------: | :-------: | :-------: | :------: |
| **Cohen kappa** |   99.23   |   87.70   |  87.42   |
|  **Precision**  |   99.39   |   90.19   |  90.05   |
|   **Recall**    |   99.38   |   90.16   |  89.94   |
|     **F1**      |   99.38   |   90.00   |  89.84   |
|  **Accuracy**   |   99.38   |   90.16   |  89.94   |
