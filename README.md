# IMDWID
IMproving Detections WIth D-RISE

Using D-RISE to interpret, understand, and improve decisions with multi-source data

Instructions:

To begin, make sure Mask R-CNN runs on your system. Follow the Matterport Mask R-CNN installation instructions; all the packages required are there. Alternatively, use the requirements file in this directory.

Run download_rgbd-objects_full.sh to get the dataset on your system. 

Next, run split_data.py to get a stratified train test split.

Next, train a model (commented with 'base' in rgbd_train) in the samples/rgbd directory. Train for 2 epochs to recreate this work.

Then, run rgbd_eval.py on your model to find which of the image ids you are missing in the train set. Copy these ids into DRISE.py (D-RISE/DRISE.py, all_ids line), and input the path to your weights in the model definition.

Go to D-RISE directory, and create a directory called 'masks'

Run python3 DRISE.py. This will take a while. In general, the s parameter should be higher if your objects are smaller, and the higher the s parameter the more masks you should use. Generally p1 is set 0.5.

Next, run rgbd_train.py uncommenting the DRISE augmentation (first model.train command). Train for 20 epochs.

Finally, run rgbd_train on the base model for 20 epochs and compare the results, being sure to check your results at 10 epochs. The DRISE augmentation method should converge faster than standard augmentation. 


