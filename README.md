# SEGAN with improved wgan

*Under construction!! Don't fork!!*

Tensorflow 1.2rc

Using imporved wgan

Enhancement on both waveform data and LPS data

Stage1 training only L1/L2 loss, without adversarial loss

Stage2 joint training

# Usage #

__Preparing data(data_utils.py)__

```python
reader = dataPreprocessor(path_to_record_name, path_to_noisy, path_to_clean, use_waveform=True)
reader.write_tfrecord()
```

__Training phase__

In main.py
change mode to:
```python
mode = 'stage1' # stage1, stage2, test
```
then
```bash
python main.py
```
__Testing phase__

In main.py
change mode to:
```python
mode = 'test' # stage1, stage2, test
```
then change test_path and test_list
```bash
python main.py
```
