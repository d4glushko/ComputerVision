# How to run
`python .\main.py --dataset "panda"`

Allowed datasets with preconfigured ROI and filters:
- panda
- basketball
- biker

Or add any other dataset in the `data` folder and add ROI and filters to the `utils.py`

# Results
As seen on the examples (basketball, biker, panda) Meanshift and Camshift performance depends heavily on the good filters to create a good mask. Otherwise they both lose the object because of similar background or other similar objects.

Meanshift has constant bounding box thus it may be bad when objects increase or decrease, but it's invariant to the rotations.

On the provided examples Meanshift performed better, as Camshift quickly lost an object.