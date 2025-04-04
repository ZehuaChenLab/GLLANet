# GLLANet
Road extraction from remote sensing images holds significant application value in various fields. However,due to the diverse shapes of roads and potential occlusions
or interferences from the background, road extraction remains a challenging task. To address these challenges, we propose a road extraction network based on 
Global-Local Linear Attention (GLLANet). First, we introduce Local Sparse Attention (LSA) using linear deformable convolution and attention weight selection to capture 
fine-grained road features. Subsequently, SS2D is adopted as Global Linear Attention (GLA) and integrated with LSA. In conjunction with a multi-scale forward feedback
network, they collectively establish the Global-Local Linear Attention Module (GLLAM). This module serves as an encoder for progressive feature extraction. 
The extracted features are passed to the decoder while preserving global-local information across different feature scales. This approach enhances road detail 
representation and provides accurate contextual information. Experiments conducted on two public road datasets demonstrate that GLLANet outperforms existing methods 
in various evaluation metrics, including Intersection over Union (IoU) and F1-score. 
