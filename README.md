# SEGAN with improved wgan

*Under construction!! Don't fork!!*

Best configuration so far!!

**On waveform**

| Metrics       | PESQ         |    SSNR    |     STOI    |
| ------------- | ------------ | ---------- | ----------- |
| Noisy         | 1.9701       | -0.0691    | 0.9210      |
| SEGAN         | 2.16         | 7.73       | X           |
| Stage1_20kep  | 2.2858       | 9.4668     | 0.9251      |
| Stage1_35kep  | 2.4063       | 9.4958     | 0.9288      |
| Joint_45kep   | 1.9877       | 8.6272     | 0.9205      |

Tensorflow 1.2rc

Using imporved wgan

Enhancement on both waveform data and LPS data

Stage1 training only L1/L2 loss, without adversarial loss

Stage2 joint training


