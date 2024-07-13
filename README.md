# LaRa: Efficient Large-Baseline Radiance Fields

[Project page](https://apchenstu.github.io/LaRa/) | [Paper](https://arxiv.org/abs/2407.04699) | [Data](https://huggingface.co/apchen/LaRa/tree/main/dataset) | [Checkpoint](https://huggingface.co/apchen/LaRa/tree/main/ckpts) |<br>

![Teaser image](assets/demo.gif)

## ⭐ New Features 
- 2024/04/05: Important updates - 
Now our method supports half precision training, achieving over **100% faster** convergence and about **1.5dB** gains with less iterations!

    | Model    | PSNR ↑     | SSIM ↑    | Abs err (Geo) ↓   | Epoch  | Time（day）      | ckpt |
    | ------   | ------     | ------    | ------    | ------ | ------ | ------ |
    | Paper    | 27.65      |  0.951    | 0.0654    |  50    |   3.5  | ------ |
    | bf16     | 29.15      |  0.956    | 0.0574    |  30    |   1.5  | [Download](https://huggingface.co/apchen/LaRa/tree/main/ckpts/) |

    Please download the pre-trained checkpoint from the provided link and place it in the `ckpts` folder.

# Installation

```
git clone https://github.com/autonomousvision/LaRa.git --recursive
conda env create --file environment.yml
conda activate lara
```


# Dataset 
We used the processed [gobjaverse dataset](https://aigc3d.github.io/gobjaverse/) for training. A download script `tools/download_dataset.py` is provided to automatically download the datasets.

```
python tools/download_dataset.py all
```
Note: The GObjaverse dataset requires approximately 1.4 TB of storage. You can also download a subset of the dataset. Please refer to the provided script for details. Please manually delete the `_temp` folder after completing the download.

If you would like to process the data by yourself, we provide preprocess scripts for the gobjaverse and co3d datasets, please check `tools/prepare_dataset_*`.
You can also download our preprocessed data and put them to `dataset` folder:
* [gobjaverse](#gobjaverse)
* [Google Scaned Object](#GSO)
* [Co3D](#Co3D) 
* Instant3D - Please contact the authors of Instant3D if you wish to obtain the data for comparison.
# Training
```
python train_lightning.py
```
**note:** You can configure the GPU id and other parameter with `configs/base.yaml`.

# Evaluation
Our method supports the reconstruction of radiance fields from **multi-view**, **text**, and **single view** inputs. We provide a pre-trained checkpoint at [ckpt](#https://huggingface.co/apchen/LaRa/tree/main/ckpts).

## multi-view to 3D
To reproduce the table results, you can simply use:
```
python eval_all.py
```
**note:** Please double-check that the paths inside the script are correct for your specific case.

## text to 3D
```
python evaluation.py configs/infer.yaml 
       infer.ckpt_path=ckpts/epoch=29.ckpt
       infer.save_folder=outputs/prompts/
       infer.dataset.generator_type=xxx
       infer.dataset.prompts=["a car made out of sushi","a beautiful rainbow fish"]
```
**note:** This part is not avariable now due the permission issue.


## single view to 3D
```
python evaluation.py configs/infer.yaml 
       infer.ckpt_path=ckpts/epoch=29.ckpt
       infer.save_folder=outputs/single-view/
       infer.dataset.generator_type="zero123plus-v1"
       infer.dataset.image_pathes=\["assets/examples/13_realfusion_cherry_1.png"\]
```
**note:** It supports the generator types `zero123plus-v1.1` and `zero123plus-v1`.



## Acknowledgements
Our render is built upon [2DGS](https://github.com/hbb1/2d-gaussian-splatting). The data preprocessing code for the Co3D dataset is partially borrowed from [Splatter-Image](https://github.com/szymanowiczs/splatter-image/blob/main/data_preprocessing/preprocess_co3d.py). Additionally, the script for generating multi-view images from text and single view image is sourced from [GRM](https://github.com/justimyhxu/grm). We thank all the authors for their great repos. 

## Citation
If you find our code or paper helps, please consider citing:
```bibtex
@inproceedings{LaRa,
         author = {Anpei Chen and Haofei Xu and Stefano Esposito and Siyu Tang and Andreas Geiger},
         title = {LaRa: Efficient Large-Baseline Radiance Fields},
         booktitle = {European Conference on Computer Vision (ECCV)},
         year = {2024}
        } 
```


