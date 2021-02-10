# VocalTechClass

`python main.py` should train the model using the attention mechanism automatically. However the attention mechanism is not complete, as I am unsure what to do with the context vector thats been generated.
The attention mechanism can be viewed in `models.py`

`python main.py --use_attention=false` should train the network and bypass the attention mechanism. However this network only scores roughly 38% accuracy, while in the original paper [Exploratory Study on Perceptual Spaces of the Singing Voice](https://ieeexplore-ieee-org.ezproxy.library.qmul.ac.uk/abstract/document/9054582), 90% accuracy is achieved.

I am currently attempting to learn why my classification score of 38% is so low, and hoping to find out if I have correctly implemented the attention mechanism. Any suggestions?

Thanks,
Brendan
