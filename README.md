# A-Navigation-System-based-on-Event-Camera-for-People-Living-with-Blindness-and-Vision-Impairment

This is the code for paper: A Navigation System based on Event Camera for People Living with Blindness and Vision Impairment. This paper proposed a proof-of-concept event-based blind navigation system.

- Download the pretrained model:

```bash
wget "http://rpg.ifi.uzh.ch/data/E2DEPTH/models/E2DEPTH_si_grad_loss_mixed.pth.tar" -O pretrained/E2DEPTH_si_grad_loss_mixed.pth.tar
```

- Required packages are listed in: requirements.txt

```bash
pip3 install -r requirement.txt
```

- Run the real-time path navigation under python file: realtime_avoidance_mt.py
