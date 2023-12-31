import torch

house_node_list_dict={
    "House_8":
    [
    torch.Tensor([-1.166203-4.65,-(2.246365+7.23)]),#1
    torch.Tensor([-1.166203-4.78,-(2.246365+0.08999991)]),#2
    torch.Tensor([-1.166203-3.97,-(2.246365+4.87)]),#3
    torch.Tensor([-1.166203-2.9,-(2.246365+2.12)]),#4
    torch.Tensor([-1.166203-0.8499999,-(2.246365+1.93)]),#5
    torch.Tensor([-1.166203+0.98,-(2.246365+0.9200001)]),#6
    torch.Tensor([-1.166203-5.31,-(2.246365-1.02)]),#7
    torch.Tensor([-1.166203-0.46,-(2.246365-1.42)]),#8

    ]
    ,
    "House_12":
    [
    torch.Tensor([-1.166203-2.791,-(2.246365-1.829)]),#1
    torch.Tensor([-1.166203-2.051063,-(2.246365-4.021405)]),#2
    torch.Tensor([-1.166203-2.494206,-(2.246365-5.704995)]),#3
    torch.Tensor([-1.166203-4.322511,-(2.246365-7.217872)]),#4
    torch.Tensor([-1.166203-2.618146,-(2.246365-7.138311)]),#5
    torch.Tensor([-1.166203+1.286348,-(2.246365-5.886983)]),#6
    torch.Tensor([-1.166203+2.02355,-(2.246365-2.612659)]),#7
    ]
    }

house_color_id_dict={
    "House_8":
    [
    [0,255,255],#1
    [0,0,255],
    [255,0,0],
    [255,0,255],
    [0,255,0],
    [255,255,0],
    [128,128,0],
    [128,0,128],#8
    [0,128,128],#9
    ]
    ,
   "House_12":
   [
   [0,255,255],         #1
   [0,0,255],
   [255,0,255],
   [255,0,0],
   [255,255,0],
   [255,128,128],
   [0,255,0]            #7
    ]
    }

house_node_toplogic_dict = {
    "House_8":
[
    [0,1,0,0,0,0,0,0,0],#1
    [1,0,1,0,0,0,0,0,1],
    [0,1,0,0,0,0,0,0,0],#3
    [0,0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,0,0,1],#5
    [0,0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,0,0,1],#7
    [0,0,0,0,0,0,0,0,1],
    [0,1,0,1,1,1,1,1,0],#9

]
    ,
    "House_12":
    [
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 1, 0],
        [1, 1, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 1, 0],
    ]
}

house_node_toplogic_edge_dict = {
    "House_8":
    [
        [0,[[121,158],[121,243]],0,0,0,0,0,0,0],#1
        [[[121,243],[121,158]],0,[[121,243],[172,243]],0,0,0,0,0,[[99,474],[141,500]]],
        [0,[[172,243],[121,243]],0,0,0,0,0,0,0],#3
        [0,0,0,0,0,0,0,0,[[238,441],[238,498]]],
        [0,0,0,0,0,0,0,0,[[381,443],[381,499]]],#5
        [0,0,0,0,0,0,0,0,[[457,497],[381,499]]],
        [0,0,0,0,0,0,0,0,[[238,556],[238,498]]],#7
        [0,0,0,0,0,0,0,0,[[381,559],[381,499]]],
        [0,[[141,500],[99,474]],0,[[238,498],[238,441]],[[381,499],[381,443]],[[381,499],[457,497]],[[238,498],[238,556]],[[381,499],[381,559]],0]#9
    ]
    ,
    "House_12":
    [
        [0, 0, 0, 0, 0, [[574,242],[574,372]], 0],
        [0, 0, 0, 0, 0, [[472,372],[574,372]], 0],
        [0, 0, 0, 0, 0, [[357,652],[460,652]], 0],
        [0, 0, 0, 0, [[137,823],[220,823]], 0, 0],
        [0, 0, 0, [[220,823],[137,823]], 0, [[459,758],[460,652]], 0],
        [[[574,372],[574,242]], [[574,372],[472,372]], [[641,652],[357,652]], 0, [[460,652],[459,758]], 0, [[574,372],[692,370]]],
        [0, 0, 0, 0, 0, [[692,370],[574,372]], 0],
    ]
}