<!--<h3><b>DPR</b></h3>-->
## <b>Deep Single Image Portrait Relighting</b> [[Project Page]](http://zhhoper.github.io/dpr.html/) <br>
[Hao Zhou], [Sunil Hadap], [Kalyan Sunkavalli], [David W. Jacobs]. In [ICCV, 2019]

<p><img src="result/obama_00.png" width="150px" >
<img src="result/obama_01.png" width="150px" >
<img src="result/obama_02.png" width="150x" >
<img src="result/obama_03.png" width="150px" >
<img src="result/obama_04.png" width="150px" >
<img src="result/obama_05.png" width="150px" >
<img src="result/obama_06.png" width="150px" >
</p>
<p><img src="result/light_00.png" width="150px" >
<img src="result/light_01.png" width="150px" >
<img src="result/light_02.png" width="150x" >
<img src="result/light_03.png" width="150px" >
<img src="result/light_04.png" width="150px" >
<img src="result/light_05.png" width="150px" >
<img src="result/light_06.png" width="150px" >
</p>

### Overview
 - (0) Test script for 512x512 images: `testNetwork_demo_512.py`
 - (1) Test script for 1024x1024 images: `testNetwork_demo_1024.py`

### Dependencies ###
pytorch > 1.0.0
opencv > 4.0.0

### Notes
We include an example image and seven example lightings in data


For real data, please download the aligned and cropped [CelebA face dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).


### Citation
If you use this code for your research, please consider citing:
```
@InProceedings{DPR,
  title={Deep Single Portrait Image Relighting},
  author = {Hao Zhou, Sunil Hadap, Kalyan Sunkavalli, David W. Jacobs},
  booktitle={International Conference on Computer Vision (ICCV)},
  year={2019}
}
```
