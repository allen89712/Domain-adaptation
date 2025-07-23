This code implements Source-Free Domain Adaptation (SFDA) using the SHOT (Source Hypothesis Transfer) framework. We adopt a ResNet-based feature extractor and replace all standard Batch Normalization (BN) layers with two alternative normalization strategies proposed in our research:

MetaBN: A meta-learning-based batch normalization mechanism that dynamically generates affine parameters to improve task-specific adaptation.

Meta-Affine: A novel module developed as part of our thesis work, which extends MetaBN by enhancing the flexibility of affine transformation for better domain alignment.

The modified SHOT pipeline aims to improve model generalization on the target domain without access to source data, by leveraging better normalization strategies under domain shift.
