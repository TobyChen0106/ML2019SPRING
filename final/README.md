# ML2019final_retinanet
## Package Installation:
Before running train.sh, please run install.sh
and download the model from drive: https://drive.google.com/file/d/1kr5NpjZEV2iZThgXn_-NFJ2u_UfBrhgq/view?fbclid=IwAR0ujRkIk60kmz9jkvZsELzsoY-PH_stLeDIMPC3iOyEwPxIL0ij0q8r2Po
## Training
For train.sh, there are some arguments that you change:
  1. --dataset: Please type **csv** here.
  2. --csv_train: Please type **The filepath of your training annotation file**. The training annotation file should have the      following format:
      
      If the patient has tumor:
        image_path1,x11,y11,x12,y12,tumor
        image_path1,x21,y21,x22,y22,tumor
        image_path2,x11,y11,x12,y12,tumor
        ......validation annotation file
      
      If not:
        image_path,,,,,
        
  3. --csv_classes: Please type **The filepath of your classes details file**. The classes details file shoud have the following format:
  
      class0,0
      class1,1
      class2,2
      
  4. --csv_val: Please type **The filepath of your validation annotation file**. The validation annotation file should have the same formate as the training annotation file.
  
  5. --model: If you already have a model file and you want to train that model again, please type **The filepath of your model file** here.
  
  6. --depth: This defines the depth of the resnet model (For ResNet, only 18, 34, 50, 101, 152 are allowed. For ResNext, only 101 is allowed)
  
  7. --resnext: If typed, the backbone model will become ResNext.
  
  8. --epochs: This defines the **number of epoch** (default=12).
  
  9. --batch_size: This defines the **batch_size** for training data (default=4).
  
  10. --workers: the number of workers(default=4).
  
  11. --lr: This defines the **learning rate** (default=1e-5).
  
  12. --angle: This defines the **rotation angle** in data augmentation (default=6).
  
  13. --zoom_range: This defines the **zoom in and zoom out ratio** in data augmentation (default=[-0.1,0.1]).
  
  14. --size: This definese the length of the pictures' sides that are passed into the model.
  
  15. --alpha: This defines **alpha in focal loss** (default=0.25).
  
  16. --gamma: This defines **gamma in focal loss** (default=2).
  
  17. --losses_with_no_bboxes: If you want to **calculate the loss on pictures without bboxes**, please type it.
  
  18. --no_bboxes_alpha: This defines **alpha in focal loss for pictures without bboxes** (default=0.5).
   
  19. --no_bboxes_gamma: This defines **gamma in focal loss for pictures without bboxes** (default=2).
  
  20. --dropout1: This defines **Dropout Rate for the 3rd cnn layer in ClassficationModel** (default=0.25).
  
  21. --dropout2: This defines **Dropout Rate for the 4th cnn layer in ClassficationModel** (default=0.25).

## Prediction
For predict.sh, there are some arguments that you change:
  1. --dataset: Please type **csv** here.
  
  2. --csv_test: Please type **The filepath of your testing metadata file**. The testing metadata file should have the following format:
    
    image_path1,Female_or_Male,AP_or_PA,age
    
  3. --csv_classes: Please type **The filepath of your classes details file**. The classes details file shoud have the same format as above.
  
  4. --model: Please type **The filepath of your prediction model** here. The prediction result will have the filename **prediction.csv**

## Ensemble
For ensemble.sh, you can change the files used in ensemble method in ensemble.py to have different results.

## Visualization
For visualize.sh, there are some arguments that you can change:
  1. --dataset: Please type **csv** here.
  
  2. --csv_val: Please type **The filepath of your testing metadata file**. The testing metadata file should have the same format as for the training case.
    
  3. --csv_classes: Please type **The filepath of your classes details file**. The classes details file shoud have the same format as above.
  
  4. --model: Please type **The filepath of your prediction model** here. The prediction result will have the filename **prediction.csv**
