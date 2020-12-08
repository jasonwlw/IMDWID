My implementation of the D-RISE Algorithm for generating saliency maps from object detection models. 

D-RISE:
      title={Black-box Explanation of Object Detectors via Saliency Maps}, 
      author={Vitali Petsiuk and Rajiv Jain and Varun Manjunatha and Vlad I. Morariu and Ashutosh Mehra and Vicente Ordonez and Kate Saenko},
      year={2020},
      eprint={2006.03204},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

RISE:

@misc{petsiuk2018rise,
      title={RISE: Randomized Input Sampling for Explanation of Black-box Models}, 
      author={Vitali Petsiuk and Abir Das and Kate Saenko},
      year={2018},
      eprint={1806.07421},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

RISE GIthub:

https://github.com/eclique/RISE

@inproceedings{Petsiuk2018rise,
  title = {RISE: Randomized Input Sampling for Explanation of Black-box Models},
  author = {Vitali Petsiuk and Abir Das and Kate Saenko},
  booktitle = {Proceedings of the British Machine Vision Conference (BMVC)},
  year = {2018}
}

Overview of the DRISE Augmentation method:

<img src="/src/DRISE_Aug.png">

The saliency maps are treated as masks; the image in the lower left is color augmented, the image in the middle is brightness augmented, and the image on the right is a combination of both augmentations. 
