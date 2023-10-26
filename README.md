# Transcribe3D

This code repository is for internal testing only, please forgive some of its imperfections:)

## Install

For evaluation, only some simple packages were used, include *numpy*, *openai* and *tenacity*.

## OpenAI API access
Set up your OpenAI api key as an environment variable `OPENAI_API_KEY`.

## Data Prepare
~~You should have a folder(let's call it *scannet_data_root*) that has ScanNet scene folders such as *scene0000_00* under it.
Besides the original ScanNet data, object bounding boxes are also needed(ground truth, group free or mask3d). Currently these boxes are directly provided. [Download](https://drive.google.com/drive/folders/1A1nV66J-8NVExauugvlc7X5FM2QhQzeW?usp=drive_link) them and unzip under *scannet_data_root*, so that there will also be these 3 folders under *scannet_data_root*: objects_info, objects_info_gf and objects_info_mask3d_200c.~~

To make things easier, we provide the bounding boexes for each scene at `data/scannet_object_info`. Currently it only include ground truth bounding boxes (which is the setting for NR3D and SR3D from Referit3D benchmark), detected bounding boxes will be provided later. There is no need to prepare the original scannet scene data for the sole purpose of testing (original scene data are still useful for debugging and visualization).

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
Implements the object filter which filters out irrelevant object according to the description.

## Evaluation
Run the first 50 data records of *nr3d_test_sampled1000.csv* with config index 1:

`python main.py --scannet_data_root /path/to/ScanNet/Data/ --script_root /path/to/Transcribe3D/project/folder --mode eval --dataset nr3d --conf_idx 1 --range 2 52`

Remember to replace the paths.

~~For the scannet_data_root, there should be scannet scene folders under it, such as *scene0000_00*~~

~~If you are using TTIC slurm, the scannet_data_root should be */share/data/ripl/scannet_raw/train/*.~~

`scannet_data_root` can be set to `/path/to/Transcribe3D/project/folder/data/scannet_object_info` as we provide the GT scannet bounding boxes already.

To run sr3d or scanrefer, simply modify the --dataset setting.

We test the first 300 samples for NR3D and 140 samples for SR3D in our paper.

After running the evaluation, a folder which has a name starting with 'eval_results_' and containing configuration infomation will be created. Under this folder, there will be subfolders named after time.

## Analyze Result
You might run one or more experiments of a evaluation configuration, and get some subfolders named after formatted time. The time/times are used for analyze the results. An example timestamp looks like "2023-10-26-15-48-12".

Specify the formatted time(s) after the --ft setting:

`python main.py --scannet_data_root /path/to/ScanNet/Data/ --script_root /path/to/Transcribe3D/project/folder --mode result --dataset nr3d --conf_idx 1 --ft time1 time2`

## Self Correction
This checks all the result dialogues given formatted time(s), self-corrects those wrong cases.

`python main.py --scannet_data_root /path/to/ScanNet/Data/ --script_root /path/to/Transcribe3D/project/folder --mode self_correct --dataset nr3d --conf_idx 1 --ft time1 time2`