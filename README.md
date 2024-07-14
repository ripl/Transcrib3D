# Transcrib3D: 3D Referring Expression Resolution through Large Language Models

[[Paper]](https://arxiv.org/abs/2404.19221) [[Website]](https://ripl.github.io/Transcrib3D/)

Authors...

Abstract...

Demos charts...

## Bibtex

```
xxx
```

## Environment Settings

For evaluation, only some simple packages are required, include *numpy*, *openai* and *tenacity*.

```bash
pip install numpy openai tenacity
```

Some additional packages are required for data preprocessing:

```bash
pip install plyfile scikit-learn scipy pandas
```

Set up your OpenAI api key as an environment variable `OPENAI_API_KEY`:

```bash
export OPENAI_API_KEY=xxx
```

## Data Prepare

Since [ReferIt3D](https://referit3d.github.io/) dataset(includes sr3d and nr3d) and [ScanRefer](https://daveredrum.github.io/ScanRefer/) dataset depend on [ScanNet](http://www.scan-net.org/), we preproces the ScanNet data as the first step.

### quick start

To make things easier, we provide the bounding boexes for each scene at `data/scannet_object_info`. Currently it only include ground truth bounding boxes (which is the setting for NR3D and SR3D from Referit3D benchmark), detected bounding boxes will be provided later. There is no need to prepare the original scannet scene data for the sole purpose of testing (original scene data are still useful for debugging and visualization). 

*You could jump to **Evaluation** to get quick start.*

If you want to generate the bounding boxes from the original ScanNet data, follow the steps below.

### download ScanNet data

Follow the official instruction [here](https://github.com/ScanNet/ScanNet) to download the data. You have to fill a form and email the authors of ScanNet. Then, you will receive a response email with detailed instructions and a download python script `download-scannet.py`. Run the script to download certain types of data:

```bash
python download-scannet.py -o [directory in which to download] --type [file suffix]
```

Since the original 1.3TB ScanNet data contains many types of data files and there are some types we don't need in this project(such as the RGBD stream `.sens` type), you could use `--type` tag to download only the necessary types:

 `_vh_clean_2.ply _vh_clean_2.labels.ply _vh_clean_2.0.010000.segs.json _vh_clean.segs.json .aggregation.json _vh_clean.aggregation.json .txt`

Run this sh/CMD instruction to download them (to avoid any key-pressing during download, comment the code `key = input('')` at line 147 and 225):

```bash
# bash
download_dir="your_scannet_download_directory"
suffixes=(
    "_vh_clean_2.ply"
    "_vh_clean_2.labels.ply"
    "_vh_clean_2.0.010000.segs.json"
    "_vh_clean.segs.json"
    ".aggregation.json"
    "_vh_clean.aggregation.json"
    ".txt"
)
for suffix in "${suffixes[@]}"; do
    python download-scannet.py -o "$download_dir" --type "$suffix"
done
```

```cmd
CMD
set download_dir="your_scannet_download_directory"
set "suffixes=_vh_clean_2.ply _vh_clean_2.labels.ply _vh_clean_2.0.010000.segs.json _vh_clean.segs.json .aggregation.json _vh_clean.aggregation.json .txt"

for %s in (%suffixes%) do (
  python download-scannet.py -o  %download_dir% --type %s
)
```

After downloading, your folder should look like:

```
your_scannet_download_directory/
|-- scans/
|   |-- scene0000_00/
|   |   |-- scene0000_00_vh_clean_2.ply
|   |   |-- scene0000_00_vh_clean_2.labels.ply
|   |   |-- scene0000_00_vh_clean_2.0.010000.segs.json
|   |   |-- scene0000_00_vh_clean.segs.json
|   |   |-- scene0000_00.aggregation.json
|   |   |-- scene0000_00_vh_clean.aggregation.json
|   |   |-- scene0000_00.txt
|   |-- scenexxxx_xx/
|   |   |-- ...
|-- scans_test/
|   |-- scene0707_00/
|   |-- ...
|-- scannetv2-labels.combined.tsv
```

### axis-align

Then, use the axis align matrices(recorded in scenexxxx_xx.txt) to transform the coordinates of vertices:

```bash
python preprocessing/align_scannet_mesh.py --scannet_download_path [your_scannet_download_directory]
``` 

### download ReferIt3D and ScanRefer data

Follow the [ReferIt3D official guide](https://referit3d.github.io/#dataset) to download `nr3d.csv`, `sr3d.csv`, `sr3d_train.csv`,  `sr3d_test.csv` and save them at `data/referit3d` folder. 

Follow the [ScanRefer official guide](https://daveredrum.github.io/ScanRefer/) to download the dataset and put them under `data/scanrefer` folder.


### generate object information

In this step, we process the ScanNet data to acquire quantitative and semantic information of objects in each scene. 

For object instance segmentation, we use either ground truth (ScanNet official) data or off-the-shelf segmentation tool ([Mask3d](https://jonasschult.github.io/Mask3D/)).

To use ground truth segmentation data, run:

```bash
python preprocessing/gen_obj_list.py --scannet_download_path [your_scannet_download_directory] --bbox_type gt
```

You can find the results in `scannet_download_path/scans/objects_info/` and `scannet_download_path/scans_test/objects_info/`.

To use Mask3D segmentation data, first follow the [Mask3D official guide](https://github.com/JonasSchult/Mask3D) to produce the instance segmentation results, then run:

```bash
python preprocessing/gen_obj_list.py --scannet_download_path [your_scannet_download_directory] \
    --bbox_type mask3d \
    --mask3d_result_path [your_mask3d_result_directory]
# Note: mask3d_result_path should look like xxx/Mask3D/eval_output/instance_evaluation_mask3d_export_scannet200_0/val/
```

You can find the results in `scannet_download_path/scans/objects_info_mask3d_200c/`.

<!-- ~~You should have a folder(let's call it *scannet_data_root*) that has ScanNet scene folders such as *scene0000_00* under it.
Besides the original ScanNet data, object bounding boxes are also needed(ground truth, group free or mask3d). Currently these boxes are directly provided. [Download](https://drive.google.com/drive/folders/1A1nV66J-8NVExauugvlc7X5FM2QhQzeW?usp=drive_link) them and unzip under *scannet_data_root*, so that there will also be these 3 folders under *scannet_data_root*: objects_info, objects_info_gf and objects_info_mask3d_200c.~~ -->


<!-- 
## File Structure

### main.py
The main part of the project is in this file, where a class called *Refer3d* is defined. Currently, it's a large class that includes data input, prompt generation, running evaluations, and analyzing results, among other things.

### config.py
This store different configurations(test modes) of evaluation.

There are 3 dictionaries inside: confs_nr3d, confs_sr3d and confs_scanrefer. Each of them contains several configurations of that dataset. The meaning of different configuration settings could be understood from the variable names.

The indices of full model(GPT-4, use principle, use code interpreter) for nr3d and sr3d is 1 and 1, respectively.

### code_interpreter.py
This define the class *CodeInterpreter* which inherits from the class *Dialogue* defined in dialogue.py. These handle the calling of openai api and implement the code interpreter.

### object_filter_gpt4.py
Implements the object filter which filters out irrelevant object according to the description. -->

## Evaluation

### quick start
Run the first 50 data records of *nr3d_test_sampled1000.csv* with config index 1:

```bash
python main.py --workspace_path /path/to/Transcribe3D/project/folder --scannet_data_root /path/to/ScanNet/Data/  --mode eval --dataset_type nr3d --conf_idx 1 --range 2 52
```

Remember to replace the paths.

Note that `scannet_data_root` can be set to `/path/to/Transcribe3D/project/folder/data/scannet_object_info` as we provide the GT scannet bounding boxes already. If you preprocess data by yourself, it can be set to `scannet_download_path/scans/objects_info/`.

### how to modify configurations

- To run our model on different refering datasets, simply modify the `--dataset_type` setting to [sr3d/nr3d/scanrefer].

- To select the evaluation range of dataset, modify the `--range` setting. For Sr3D and Nr3D which use .csv files, the minimum number is 2. For ScanRefer which uses .json files, the minimum number is 0.

- For convenience, more configurations are put into `config/config.py`. There are 3 dictionaries inside: confs_nr3d, confs_sr3d and confs_scanrefer. Each of them contains several configurations of that dataset. The meaning of different configurations could be understood from the variable names. Modify the `--conf_idx` setting to select configuration, and you can also add your own configurations.

- More information could be found by `python main.py -h`.

### result storage

After running the evaluation with certian configuration, a folder which has a name starting with `eval_results_` and containing configuration infomation will be created under the `results` folder. Under this folder, there will be subfolders named after the start time of experiment.

## Analyze Result
You might run one or more experiments of a evaluation configuration, and get some subfolders named after formatted time. The time/times are used for analyze the results. An example timestamp looks like `2023-10-26-15-48-12`.

Specify the formatted time(s) after the `--ft` setting:

```bash
python main.py --workspace_path /path/to/Transcribe3D/project/folder/ --scannet_data_root /path/to/ScanNet/Data/  --mode result --dataset_type nr3d --conf_idx 1 --ft time1 time2
```

<!-- ## Self Correction

TODO

This checks all the result dialogues given formatted time(s), self-corrects those wrong cases.

```bash
python main.py --workspace_path /path/to/Transcribe3D/project/folder/ --scannet_data_root /path/to/ScanNet/Data/ --mode self_correct --dataset_type nr3d --conf_idx 1 --ft time1 time2
``` -->

## Check Scanrefer

Check the how many cases are provided with detected boxes that has 0.5 or higher iou with gt box, which indicates the upper bound of performance on Scanrefer.

```bash
python main.py --workspace_path /path/to/Transcribe3D/project/folder/ --scannet_data_root /path/to/ScanNet/Data/ --mode check_scanrefer --dataset_type scanrefer --conf_idx 1
```

## Finetuning
We provide scripts for finetuning on open-source LLMs (e.g. codeLlama, Llama2) under `finetune` directory.

### Environment
The script uses Huggingface `trl` (https://github.com/huggingface/trl) library to perform finetuning jobs. Main dependencies include Huggingface `accelerate`, `transformers`, `datasets`, `peft`, `trl`.

### Data
We provide processed fientuning data following OpenAI finetune file protocal in `finetune/finetune_files` directory. It contains many different settings aligned as described in our paper. The original processing script is `finetune/prepare_finetuning_data.py` which processes results from the main script.

### Scripts
We provide two example shell scripts to run the finetuning jobs, one with `codellama` model (`finetune/trl_finetune_codellama_instruct.sh`) and the other with `llama2_chat` model (`finetune/trl_finetune_llama2_chat.sh`). You can also customize finetuning job using `finetune/trl_finetune.py`.

### Notes
- The finetuned open-source models (e.g. codellama, llama2) still large under performs finetuned closed-source model (gpt-3.5-turbo) as of Sep 2023. We expect the situation would change dramatically in the coming future with quickly improving open source models.
- The resource required for finetuning is roughly 24g+ gpu memory for 7b models and 36g+ gpu memory for 13b models.



