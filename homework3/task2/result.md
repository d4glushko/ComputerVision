# How to run
`python .\main.py --dataset "panda"`

Allowed datasets with preconfigured ROI and filters:
- panda
- basketball
- biker

Or add any other dataset in the `data` folder and add ROI and filters to the `utils.py`

# Results
Lucas-Kanade works pretty good and fast on tracking points, as it find the best guess in the neighbour window. It works on moderate object speeds and shows good results. However in some cases it loses the object:
 - when the objects overlap each other, then algorithm will start tracking the above object (example with 'basketball' and 'panda' datasets)
 - when the speed is too fast, as it will not find the corresponding point on the next frame (example with the 'biker')

 Extension with pyramids level improves the overall quality and shows better results in the mentioned problems and datasets: algorithm didn't lose the biker.