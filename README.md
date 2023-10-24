# Transcribe3D

This code repository is for internal testing only, please forgive its imperfections.

## Install

For evaluation, only some simple packages were used, include *numpy*, *openai* and *tenacity*.

## File Structure

### main.py
The main part of the project is in this file, where a class called *Refer3d* is defined. Currently, it's a large class that includes data input, prompt generation, running evaluations, and analyzing results, among other things.

### config.py
This store different configurations(test modes) of evaluation.

### code_interpreter.py
This define the class *CodeInterpreter* which inherits from the class *Dialogue* defined in dialogue.py. These handle the calling of openai api and implement the code interpreter.

### object_filter_gpt4.py
Implements the object filter which filters out irrelevant object according to the description.

## Evaluation
