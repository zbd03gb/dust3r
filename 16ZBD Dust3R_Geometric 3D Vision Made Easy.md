[dust3R.pdf](https://www.yuque.com/attachments/yuque/0/2025/pdf/58377837/1766395163003-dd6f8acc-f693-4c62-b7eb-4814055a4f9f.pdf)

**Github:** [https://github.com/naver/dust3r](https://github.com/naver/dust3r)

# 1 Introduction
## 1.1 研究背景
三维几何感知是计算机视觉领域中的核心问题之一，是机器人导航、增强现实、三维重建以及自动驾驶等任务的基础。长期以来，三维视觉问题通常依赖于明确的几何建模与精确的相机参数，通过多视几何理论将二维图像信息恢复为三维结构。经典方法如 SfM 和 MVS 通常采用“**特征匹配—相机位姿估计—三角化—全局优化**”的流水线式框架，在受控条件和精确标定的情况下可以取得较高精度。然而，**这类方法对相机内外参、特征可重复性以及几何假设具有较强依赖，在真实复杂场景中往往表现出较差的鲁棒性**。

近年来，深度学习方法在视觉领域取得了显著进展，研究者开始尝试用神经网络替代或弱化传统几何模块，例如学习式特征匹配、单目或多目深度估计，以及端到端位姿回归等。这些方法在一定程度上缓解了传统几何方法对人工设计模块的依赖，**但大多数仍然建立在明确的相机模型与几何约束之上，或需要复杂的后处理步骤来保证几何一致性**。

在这一背景下，DUSt3R 提出了一种新的视角，尝试从根本上重新思考三维视觉问题的建模方式。**作者提出，不再将三维重建视为由多个几何子问题拼接而成的过程，而是直接学习一个能够从图像对中预测三维结构的模型，从而避免对显式相机参数和几何推导的依赖。**



## 1.2 相关工作
三维视觉领域的相关研究主要可以分为基于几何的传统方法和基于学习的直接预测方法。运动结构恢复（Structure-from-Motion, SfM）[20,21,45] 旨在从多张图像中联合估计相机参数并重建稀疏三维结构，**其经典流程依赖于关键点匹配获取像素对应关系，再通过几何约束和 bundle adjustment 联合优化三维点和相机参数。**近年来**，SfM 的各个子模块逐渐引入学习方法，包括特征描述[28]、图像匹配[60]、特征度量细化[59]以及神经化的 bundle adjustment[153]**，使得整体性能有所提升。然而，**该方法仍然保持级联流水线结构，使其对单个模块中的噪声和误差较为敏感。**

> **[20] Crandall et al., **_**SfM with MRFs**_**, PAMI 2013**:
>
> 该工作将大规模结构恢复问题建模为离散–连续联合优化，引入马尔可夫随机场（MRF）以增强全局一致性，从而提升了 SfM 在复杂场景中的鲁棒性。
>
> **[21] Cui et al., **_**HSfM: Hybrid Structure-from-Motion**_**, CVPR 2017**:
>
> HSfM 提出一种混合式 SfM 框架，将全局方法的稳定性与增量式方法的效率相结合，在保证重建质量的同时显著提升了可扩展性。
>
> **[28] Dusmanu et al., **_**D2-Net**_**, CVPR 2019**：
>
> D2-Net 使用单一 CNN 同时完成特征检测和描述，通过密集特征图中的极值选择关键点，实现了可训练的联合检测与描述框架。 
>
> **[45] Hartley and Zisserman, **_**Multiple View Geometry in Computer Vision**_**, 2004**：
>
> 该著作系统性地奠定了多视几何的理论基础，涵盖相机模型、极线几何、三角化与三维重建等核心内容，是 SfM 和 MVS 领域的经典参考文献。
>
> **特征度量细化(featuremetric refinement)**：
>
> 在不改变传统 SfM 流水线结构的前提下，用“特征空间误差”替代纯像素重投影误差，对相机位姿和结构进行精细化优化，从而实现亚像素级的重建精度。
>
> **[59] Lindenberger et al., **_**Pixel-perfect Structure-from-Motion with Featuremetric Refinement**_**, ICCV 2021**  
该工作在传统 SfM 流水线中引入特征度量空间中的连续优化，通过 featuremetric refinement 提升像素级对齐精度，从而显著改善相机位姿与三维重建质量。
>
> **[60] Lindenberger et al., **_**LightGlue**_**, ICCV 2023**：
>
> LightGlue 是一种高效的学习式特征匹配框架，通过自适应计算和早停机制，在保持匹配精度的同时显著提升了推理速度。  
>
> **[153] Xiao et al., **_**LevelS2fM: Structure From Motion on Neural Level Set of Implicit Surfaces**_**, CVPR 2023**  
该工作在隐式曲面表示的 level set 上执行 SfM，通过联合优化相机参数和隐式几何，实现了在神经表示空间中的端到端结构恢复。  
>

**多视图立体（Multi-View Stereo, MVS）关注于从多视角图像中密集重建可见表面。这是通过多个视点之间的三角测量来实现的。**在 MVS 的经典公式中，所有相机参数都应该作为输入提供。传统 [147]、优化型[31]以及学习型 [71]MVS 方法**普遍假设相机参数已知**，并依赖复杂的相机标定或 SfM 结果。**在真实场景中，相机参数估计的不准确性往往会显著影响重建质量。针对这一问题，本文提出的方法选择绕开显式相机建模，直接预测可见表面的三维几何。** 

> **[31]** _**Qiancheng Fu, Qingshan Xu, and Wenbing Tao. Geo-neus ,Geo-Neus**_** ,NeurIPS, 2022  **
>
> 提出通过几何一致性约束的神经隐式表面学习方法，实现多视图下更精确的表面重建。  
>
> **[71] Meng et al., **_**NEAT: Learning Neural Implicit Surfaces with Arbitrary Topologies from Multiview Images**_**, CVPR 2023**  
NEAT 提出从多视角图像中学习隐式表面表示的方法，能够重建具有任意拓扑结构的三维几何，但依赖已知或可估计的相机参数进行多视角约束。
>
> **[147] Wang et al., **_**Adaptive Patch Deformation for Textureless-Resilient MVS**_**, CVPR 2023**  
该工作针对低纹理区域提出自适应 patch 形变策略，有效缓解了传统 MVS 在纹理缺失场景下重建不稳定的问题。   
>

近年来也出现了直接从单张 RGB 图像预测三维几何的研究。单目 RGB-to-3D 方法通常依赖神经网络从大规模数据中学习强先验，以缓解问题本身的欠定性。其中一类方法利用类别级 [85] 对象先验进行物体级三维重建，**但难以泛化到未知类别**；另一类方法侧重于通用场景，通常基于单目深度估计网络，并结合相机内参恢复三维结构[151]。如果没有相机内在功能，一种解决方案是通过利用视频帧中的时间一致性来推断它们，或者通过强制全局对齐等。 [156] 或利用具有光度重建损失的可微分渲染 [117]。另一种方法是显式学习预测相机内在特性，这使得与 MDE 结合时能够从单个图像执行度量 3D 重建 [168]。**然而，这类方法要么依赖已知或预测的相机参数，要么受限于深度估计质量的限制，这对于单目设置来说可能是不合适的。**

> **[85] Pavllo et al., **_**Shape, Pose, and Appearance from a Single Image via Bootstrapped Radiance Field Inversion**_**, CVPR 2023**  
该工作针对单张 RGB 图像，通过辐射场反演同时恢复物体的形状、姿态和外观，依赖类别级或实例级先验，在受限对象分布下表现出较强的重建能力。
>
> **[117] Spencer et al., **_**Kick Back & Relax: Learning to Reconstruct the World by Watching SlowTV**_**, ICCV 2023**  
该工作通过长时间、缓慢变化的视频序列，利用可微渲染和光度一致性约束，在弱监督条件下学习三维场景重建，但仍依赖投影模型和时序一致性假设。  
>
> **[151] Wiles et al., **_**SynSin: End-to-End View Synthesis from a Single Image**_**, CVPR 2020**  
SynSin 基于单目深度预测和已知相机参数，从单张图像进行新视角合成，将深度图作为中间三维表示，但其几何质量受限于单目深度估计的准确性。
>
> **[156] Xu et al., **_**FrozenRecon: Pose-Free 3D Scene Reconstruction with Frozen Depth Models**_**, ICCV 2023**  
FrozenRecon 利用冻结的单目深度模型，在无需相机位姿监督的情况下进行三维场景重建，通过多视角一致性约束间接推断几何结构。
>
> **[168] Yin et al., **_**Metric3D: Towards Zero-Shot Metric 3D Prediction from a Single Image**_**, ICCV 2023**  
Metric3D 旨在从单张 RGB 图像中实现零样本的度量级三维预测，通过大规模数据训练提升单目深度在不同场景下的泛化能力。  
>

过去已经提出了用于 3D 重建的多视图网络。它们本质上基于构建可微分 SfM 管道的理念，复制传统管道，但对其进行端到端训练 [131]。然而为此需要地面实况相机内在函数作为输入，输出通常是深度图和相对相机姿势 [184]。与这些方法不同，本文提出的网络采用通用架构，直接回归像素对齐的三维点图（pointmaps），在隐式建模相机位姿的同时，使回归问题本身更加适定。尽管 pointmap 作为形状表示对于 MVS 来说是违反直觉，但其在视觉定位被广泛应用，验证了以视角对齐三维表示在图像空间中进行建模的有效性。

> **[131] Teed and Deng, **_**DeepV2D: Video to Depth with Differentiable Structure from Motion**_**, ICLR 2020**  
DeepV2D 将可微分的 SfM 融入深度学习框架中，通过视频序列联合优化深度和相机位姿，实现端到端的多视角深度估计。
>
> **[184] Zhou et al., **_**DeepTAM: Deep Tracking and Mapping with Convolutional Neural Networks**_**, IJCV 2020**  
DeepTAM 将传统的 Tracking-and-Mapping 框架与卷积神经网络结合，在视频序列中联合学习相机跟踪和稠密深度映射，依赖显式相机几何进行多视角约束。    
>

# 1.3 创新点
+ 提出了第一个从未标定和未定位的图像中进行整体的端到端的三维重建管道，统一了单目和双目三维重建。
+ 引入了适用于MVS应用的pointmap表示，使得网络能够在一个规范的框架中预测三维形状，同时保留像素与场景之间的隐式关系。这有效地降低了通常透视相机公式的许多限制。
+ 在多视图三维重建的背景下，介绍了一种全局对齐点映射的优化过程。从某种意义上说，我们的方法统一了所有的3D视觉任务，并且在很大程度上简化了传统的重建管道，使得DUSt3R显得简单和容易。

# 2 Method
**pointmap：**

代表二维图片的像素点与其3D场景点的一一对应关系。即$ I_{i,j} \leftrightarrow X_{i,j} $, 对于每个2d像素点，都有与其对应的3d点。以（B,H,W,3）形式存储。



**cameras and scene: **

给定相机内参K与真实深度图depthmap D ,可以直接得到观测场景的pointmap：

$ X_{i,j} = K^{-1}{[iD_{i,j},jD_{i,j},D_{i,j}]}^T $

上式X表示在相机坐标系下，我们用上标$ X^{n,m} $表示相机n在相机相机m的点映射：

$ X^{n,m} = P_{m}P_{n}^{-1}h(X^n) $

其中$ P_m，P_n是世界系到m,n相机系的变换矩阵; h(X):(x,y,z)\rightarrow (x,y,z,1) $

 

## 2.1 网络架构
<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2025/png/58377837/1766293560616-b4619e54-9c42-491e-8630-827dee8d7433.png)

Dust3R通过直接回归的方式，构建网络来解决三维重建任务。网络输入两张RGB图片，输出对应的pointmap与置信度。需要注意的是，两张点图均表达在第一张图像的坐标系下。

Dust3R网络F的架构受到了CroCo [150] 的启发，使得它可以直接从CroCo预训练中获益，其网络初始化权重来自Croco。如图2所示，它由两个相同的分支(每幅图像一幅)组成，每个分支包括一个图像编码器，一个解码器和一个回归头。两个输入图像首先通过相同的权重共享ViT编码器以孪生网络的方式进行编码，产生两个token表示F1和F2。

网络在解码器中对两者进行联合推理。与CroCo类似，解码器是一个具有交叉注意力的通用transform网络。

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2025/png/58377837/1766293493529-c7510e35-b612-4ba5-836d-906a6eacc35a.png)

每个解码器块关注来自另一个分支的token：解码器模块(一共有B块)依次执行自注意力(每个视图的标记都关注同一个视图的标记)，然后交叉注意力(一个视图的每个标记都关注另一个视图的所有其它标记)，最后将令牌反馈给MLP。重要的是，在解码器传递过程中，两个分支之间的信息是不断共享的。这对于输出正确对齐的点图是至关重要的。

$ G_i^1 = DecoderBlock_i^1(G_{i-1}^1,G_{i-1}^2), $

$ G_i^2 = DecoderBlock_i^1(G_{i-1}^2,G_{i-1}^1),
 $

初始值$ G_0^1 = F_1,G_0^2 = F_2 $

最后，每个分支的回归头取解码器的token合集，输出相机1系下的pointmap1、相机2对齐相机1系后的pointmap2与对应的置信度图。

$ X^{1,1},C^{1,1} = Head^1(G_0^1,...,G_B^1) $

$ X^{2,1},C^{2,1} = Head^2(G_0^2,...,G_B^2) $

> **[150] CroCo（Cross-view Completion）** 是一种 **自监督预训练范式**，旨在让视觉模型学习 **跨视图几何关系**。它的核心预训练任务是：输入一对同一场景但不同视角的图像，随机遮挡第一张图的一部分。模型必须基于未遮挡部分和第二张完整视图来重建被遮挡部分。
>
> 因此 CroCo 与传统的 **单视图 Masked Image Modeling (MIM)** 不同，它引入第二视图作为条件，从而迫使模型 **学习不同视角之间的空间几何联系**。
>

## 2.2 训练策略
### 2.2.1 3D回归损失（3D Regression loss）
唯一的训练目标是基于3D空间中的回归。网络通过最小化预测pointmap与真实三维点之间的距离来进行优化。为缓解尺度不确定性问题，论文在损失函数中引入了尺度归一化，使网络关注于几何结构的一致性而非绝对尺度。

$ l_{regr}(v,i) = ||\frac{1}{z}X_i^{v,1} - \frac{1}{\overline z}\overline X_i^{v,1}|| $

其中，$ z = norm(X^{1,1},X^{2,1}), \overline z = norm(\overline X^{1,1},\overline X^{2,1}),norm(X^1,X^2)=\frac{1}{|D^1|+|D^2|}\sum_{v\in{1,2}}\sum_{i\in D^v}||X_i^v|| $

> GT pointmap使用$ X_{i,j} = K^{-1}{[iD_{i,j},jD_{i,j},D_{i,j}]}^T $，$ X_{i,j} = K^{-1}{[iD_{i,j},jD_{i,j},D_{i,j}]}^T $从GT 深度图、相机内参和相机姿态得出
>

```python
class Regr3D(Criterion, MultiLoss):
    """
    核心的 3D 回归损失函数，确保所有 3D 点都是正确的
    
    非对称损失：view1 被视为参考坐标系（anchor）
    
    原理：
    P1 = RT1 @ D1  # view1 的 3D 点
    P2 = RT2 @ D2  # view2 的 3D 点
    loss1 = (I @ pred_D1) - (RT1^-1 @ RT1 @ D1)  # view1 损失
    loss2 = (RT21 @ pred_D2) - (RT1^-1 @ P2)     # view2 损失
           = (RT21 @ pred_D2) - (RT1^-1 @ RT2 @ D2)
    """

    def __init__(self, criterion, norm_mode='avg_dis', gt_scale=False):
        """
        Args:
            criterion: 基础损失函数（如 L21Loss）
            norm_mode: 点云归一化模式，如 'avg_dis'
            gt_scale: 是否使用 GT 尺度，False 时预测和 GT 都归一化
        """
        super().__init__(criterion)
        self.norm_mode = norm_mode  # 点云归一化模式
        self.gt_scale = gt_scale    # GT 尺度标志

    def get_all_pts3d(self, gt1, gt2, pred1, pred2, dist_clip=None):
        """
        获取所有 3D 点，进行坐标系转换和归一化
        
        Args:
            gt1, gt2: 两个视图的 GT 数据
            pred1, pred2: 两个视图的预测数据
            dist_clip: 距离裁剪阈值，过远的点视为无效
        
        Returns:
            tuple: (gt_pts1, gt_pts2, pred_pts1, pred_pts2, valid1, valid2, monitoring)
        """
        # 所有坐标都相对于 view1 的相机坐标系进行归一化
        in_camera1 = inv(gt1['camera_pose'])  # view1 相机位姿的逆
        gt_pts1 = geotrf(in_camera1, gt1['pts3d'])  # B,H,W,3 - view1 的 GT 3D 点
        gt_pts2 = geotrf(in_camera1, gt2['pts3d'])  # B,H,W,3 - view2 的 GT 3D 点

        # 复制有效掩码
        valid1 = gt1['valid_mask'].clone()
        valid2 = gt2['valid_mask'].clone()

        # 距离裁剪：过远的点视为无效
        if dist_clip is not None:
            dis1 = gt_pts1.norm(dim=-1)  # (B, H, W) - view1 点的距离
            dis2 = gt_pts2.norm(dim=-1)  # (B, H, W) - view2 点的距离
            valid1 = valid1 & (dis1 <= dist_clip)
            valid2 = valid2 & (dis2 <= dist_clip)

        # 获取网络预测的 3D 点
        pr_pts1 = get_pred_pts3d(gt1, pred1, use_pose=False)  # view1 不使用位姿
        pr_pts2 = get_pred_pts3d(gt2, pred2, use_pose=True)   # view2 使用位姿

        # 点云归一化（解决尺度歧义问题）
        if self.norm_mode:
            pr_pts1, pr_pts2 = normalize_pointcloud(pr_pts1, pr_pts2, self.norm_mode, valid1, valid2)
        if self.norm_mode and not self.gt_scale:
            # 如果不使用 GT 尺度，也对 GT 进行归一化
            gt_pts1, gt_pts2 = normalize_pointcloud(gt_pts1, gt_pts2, self.norm_mode, valid1, valid2)

        return gt_pts1, gt_pts2, pr_pts1, pr_pts2, valid1, valid2, {}

    def compute_loss(self, gt1, gt2, pred1, pred2, **kw):
        """
        计算 3D 回归损失
        
        Returns:
            tuple: (sum_loss, details)
        """
        gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, monitoring = \
        self.get_all_pts3d(gt1, gt2, pred1, pred2, **kw)

        # view1 侧的损失
        l1 = self.criterion(pred_pts1[mask1], gt_pts1[mask1])
        # view2 侧的损失
        l2 = self.criterion(pred_pts2[mask2], gt_pts2[mask2])

        self_name = type(self).__name__
        details = {
            self_name + '_pts3d_1': float(l1.mean()),  # view1 损失均值
            self_name + '_pts3d_2': float(l2.mean())   # view2 损失均值
        }
        
        return Sum((l1, mask1), (l2, mask2)), (details | monitoring)
```

### 2.2.2 置信度感知损失（Confidence-aware loss）
在现实中，存在定义不明确的3D点，例如在天空或半透明物体上。更一般来说，图像中的某些部分通常更难预测其3d点。因此，我们联合学习为每个像素预测一个分数，该分数代表了网络对这个特定像素的信心。

$ l_{conf} = \sum_{v\in {1,2}}\sum_{i\in D_v}C_i^{v,1}l_{regr}(v,i)-\alpha logC_i^{v,1} $

$ C_i^{v,i} $为像素i的置信度，$ \alpha $为控制正则化项的超参数。为了保证严格正的置信度，通常定义$ C_i^{v,1} = 1 + exp \widetilde {C_i^{v,1}} > 1 $。这具有迫使网络在更困难的区域进行外推的效果。



```python
# 通过训练命令--train_criterion设置
--train_criterion "ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)"
```

```python
class ConfLoss(MultiLoss):
    """
    基于学习到的置信度进行加权回归的损失函数
    假设输入的 pixel_loss 是像素级回归损失
    
    原理：
        高置信度：conf = 0.1  => conf_loss = x/10 + alpha*log(10)
        低置信度：conf = 10   => conf_loss = x*10 - alpha*log(10)
        
        alpha: 超参数，控制置信度的影响程度
    """

    def __init__(self, pixel_loss, alpha=1):
        """
        Args:
            pixel_loss: 像素级损失函数
            alpha: 置信度权重超参数，必须 > 0
        """
        super().__init__()
        assert alpha > 0
        self.alpha = alpha
        self.pixel_loss = pixel_loss.with_reduction('none')  # 获取像素级损失

    def get_name(self):
        """获取损失名称"""
        return f'ConfLoss({self.pixel_loss})'

    def get_conf_log(self, x):
        """
        获取置信度及其对数值
        
        Args:
            x: 置信度张量
        
        Returns:
            tuple: (confidence, log_confidence)
        """
        return x, torch.log(x)

    def compute_loss(self, gt1, gt2, pred1, pred2, **kw):
        # 计算逐像素损失
        ((loss1, msk1), (loss2, msk2)), details = self.pixel_loss(gt1, gt2, pred1, pred2, **kw)
        
        # 检查有效像素数量
        if loss1.numel() == 0:
            print('NO VALID POINTS in img1', force=True)
        if loss2.numel() == 0:
            print('NO VALID POINTS in img2', force=True)

        # 按置信度加权
        conf1, log_conf1 = self.get_conf_log(pred1['conf'][msk1])
        conf2, log_conf2 = self.get_conf_log(pred2['conf'][msk2])
        
        # 置信度加权损失：conf * loss - alpha * log(conf)
        conf_loss1 = loss1 * conf1 - self.alpha * log_conf1
        conf_loss2 = loss2 * conf2 - self.alpha * log_conf2

        # 均值化 + NaN 保护（处理完全没有有效像素的情况）
        conf_loss1 = conf_loss1.mean() if conf_loss1.numel() > 0 else 0
        conf_loss2 = conf_loss2.mean() if conf_loss2.numel() > 0 else 0

        return conf_loss1 + conf_loss2, dict(
            conf_loss_1=float(conf_loss1), 
            conf_loss2=float(conf_loss2), 
            **details
        )
```



### 2.2.3 输入图像与相关输出
<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2025/png/58377837/1766287293324-bee4141e-d545-4af3-856f-f6727766ff06.png)

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2025/png/58377837/1766317163383-e24642bd-0e8e-4b4b-8648-abe41d11ba9d.png)

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2025/png/58377837/1766287555624-f0380659-6eac-4d03-b4a9-ade0513f0474.png)       <!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2025/png/58377837/1766287588267-d6ed56a6-2f28-4c06-95c3-a8f4e3a522eb.png)

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2025/png/58377837/1766287704140-7cbbd466-d55c-49cf-a6ab-ae9e66e5ed8b.png)        <!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2025/png/58377837/1766287736413-d469f531-7ae1-4072-ae1c-ed1fa62393d1.png)

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2025/png/58377837/1766287890074-8820a622-698c-4105-9c84-d9f736315883.png)



## 2.3 下游应用
输出点图的丰富属性使得我们可以相对轻松地执行各种便捷的操作。比如点匹配、恢复相机内参、相对位姿估计等。

### 2.3.1 Point matching 
建立两幅图像像素之间的对应关系可以通过在3D点图空间中的最近邻( NN )搜索来实现。为了最小化误差，我们通常保留图像$ I_1 $和$ I_2 $之间的互逆(相互)对应关系$ M_{1,2} $:

$ M_{1,2} = \left\{(i,j)|i=NN_1^{1,2}(j)\ and \ j=NN_1^{2,1}(i)\right\} $

$ with NN_k^{n,m}(i) = \mathop{argmin}\limits_{j \in\left\{0,...,WH\right\}}||X_j^{n,k}-X_i^{m,k}|| $

```python
def find_reciprocal_matches(P1, P2):
    """寻找相互最近邻匹配
    
    基于双向最近邻搜索的鲁棒匹配方法
    
    Args:
        P1 (np.ndarray): 第一个点集，形状 (N1, 3)
        P2 (np.ndarray): 第二个点集，形状 (N2, 3)
    
    Returns:
        tuple: (reciprocal_in_P2, nn2_in_P1, num_matches)
            - reciprocal_in_P2: P2 中的相互匹配掩码，形状 (N2,)
            - nn2_in_P2: P2 中每个点在 P1 中的最近邻索引，形状 (N2,)
            - num_matches: 相互匹配的总数量
            
    Algorithm:
        1. 在 P2 中查找 P1 的最近邻
        2. 在 P1 中查找 P2 的最近邻  
        3. 保留相互最近邻的匹配
        
    Robustness:
        - 双向检查减少错误匹配
        - 基于几何距离而非描述子
        - 适用于点云配准和验证
        
    Applications:
        - 点云配准质量评估
        - 重叠区域检测
        - 3D 对应关系验证
    """
    # 构建 KDTree 用于快速最近邻搜索
    tree1 = KDTree(P1)  # 在 P1 中搜索
    tree2 = KDTree(P2)  # 在 P2 中搜索

    # 双向最近邻搜索
    _, nn1_in_P2 = tree2.query(P1, workers=8)  # P1 中每个点在 P2 中的最近邻
    _, nn2_in_P1 = tree1.query(P2, workers=8)  # P2 中每个点在 P1 中的最近邻

    # 检查相互最近邻关系
    reciprocal_in_P1 = (nn2_in_P1[nn1_in_P2] == np.arange(len(nn1_in_P2)))  # P1 的相互匹配
    reciprocal_in_P2 = (nn1_in_P2[nn2_in_P1] == np.arange(len(nn2_in_P1)))  # P2 的相互匹配
    
    # 确保双向一致性
    assert reciprocal_in_P1.sum() == reciprocal_in_P2.sum()
    return reciprocal_in_P2, nn2_in_P1, reciprocal_in_P2.sum()
```



### 2.3.2 Recovering intrinsics
根据定义，点映射$ X^{1,1} $表示在$ I_1 $的坐标框架中。因此，可以通过求解一个简单的优化问题来估计摄像机的内参数。在这项工作中，我们假设主点是近似中心的，像素是正方形的，

$ f_1^* = \mathop{argmin}\limits_{f_1}\sum_{i=0}^{W}\sum_{j=0}^{H}C_{i,j}^{1,1}||(i^{'},j^{'})-f_1\frac{(X_{i,j,0}^{1,1},X_{i,j,1}^{1,1})}{X_{i,j,2}^{1,1}}|| $

其中，$ i^{'} = i - \frac{W}{2},j^{'} = j - \frac{H}{2} $



```python
def estimate_focal(pts3d_i, pp=None): 
    # 没有显示给定主点就使用默认假设
    if pp is None:  
        H, W, THREE = pts3d_i.shape  
        assert THREE == 3
        # 主点预设为高宽的一半
        pp = torch.tensor((W/2, H/2), device=pts3d_i.device)  
    focal = estimate_focal_knowing_depth(pts3d_i.unsqueeze(0), pp.unsqueeze(0), focal_mode='weiszfeld').ravel()  
    return float(focal



def estimate_focal_knowing_depth(pts3d, pp, focal_mode='median', min_focal=0., max_focal=np.inf):
    """ Reprojection method, for when the absolute depth is known:
        1) estimate the camera focal using a robust estimator
        2) reproject points onto true rays, minimizing a certain error
    """
    B, H, W, THREE = pts3d.shape
    assert THREE == 3

    # centered pixel grid
    pixels = xy_grid(W, H, device=pts3d.device).view(1, -1, 2) - pp.view(-1, 1, 2)  # 转到光心坐标。B,HW,2
    pts3d = pts3d.flatten(1, 2)  # (B, HW, 3)
    # 中位数估计
    if focal_mode == 'median':
        with torch.no_grad():
            # direct estimation of focal
            u, v = pixels.unbind(dim=-1)
            x, y, z = pts3d.unbind(dim=-1)
            fx_votes = (u * z) / x
            fy_votes = (v * z) / y

            # assume square pixels, hence same focal for X and Y
            f_votes = torch.cat((fx_votes.view(B, -1), fy_votes.view(B, -1)), dim=-1)
            focal = torch.nanmedian(f_votes, dim=-1).values
    # 鲁棒迭代估计
    elif focal_mode == 'weiszfeld':
        # init focal with l2 closed form
        # we try to find focal = argmin Sum | pixel - focal * (x,y)/z|
        xy_over_z = (pts3d[..., :2] / pts3d[..., 2:3]).nan_to_num(posinf=0, neginf=0)  # homogeneous (x,y,1)

        dot_xy_px = (xy_over_z * pixels).sum(dim=-1)
        dot_xy_xy = xy_over_z.square().sum(dim=-1)

        focal = dot_xy_px.mean(dim=1) / dot_xy_xy.mean(dim=1)

        # iterative re-weighted least-squares
        for iter in range(10):
            # re-weighting by inverse of distance
            dis = (pixels - focal.view(-1, 1, 1) * xy_over_z).norm(dim=-1)
            # print(dis.nanmean(-1))
            w = dis.clip(min=1e-8).reciprocal()
            # update the scaling with the new weights
            focal = (w * dot_xy_px).mean(dim=1) / (w * dot_xy_xy).mean(dim=1)
    else:
        raise ValueError(f'bad {focal_mode=}')

    focal_base = max(H, W) / (2 * np.tan(np.deg2rad(60) / 2))  # size / 1.1547005383792515
    focal = focal.clip(min=min_focal*focal_base, max=max_focal*focal_base)
    # print(focal)
    return focal
```



## 2.3.3 Relative pose estimation
可以通过几种方法实现。一种方法是进行2D匹配并恢复如上所述的内参，然后估计极线矩阵并恢复相对位姿。另一种更直接的方式是利用Procrustes对齐[64]pointmap$ X^{1,1} \leftrightarrow X^{1,2} $:

$ R^*,t^* = \mathop{argmin}\limits_{\sigma,R,t}\sum_iC_i^{1,1}C_i^{1,2}||\sigma(RX_i^{1,1}+t)-X_i^{1,2}||^2 $ 

遗憾的是，Procrustes对齐对噪声和异常值非常敏感。

> [64] Bin Luo and Edwin R. Hancock. Procrustes alignment with the EM algorithm. In Computer Analysis of Images and Patterns, CAIP, volume 1689 of Lecture Notes in Computer Science, pages 623–631. Springer, 1999. 5: 在**已知两组一一对应的点**的情况下，寻找一个**最优的刚体（或相似）变换**，使得一组点在经过该变换后与另一组点尽可能对齐，其优化目标通常是**最小化两组点之间的平方距离之和**。  
>

```python
# 刚性点配准恢复相对位姿
def rigid_points_registration(pts1, pts2, conf):  
    # 使用ROMA库进行刚性点配准，返回尺度、旋转和平移  
    R, T, s = roma.rigid_points_registration(  
        pts1.reshape(-1, 3),   
        pts2.reshape(-1, 3),   
        weights=conf.ravel(),   
        compute_scaling=True  
    )  
    return s, R, T  # 返回尺度、旋转矩阵和平移向量

# 位姿变换矩阵构建
def sRT_to_4x4(scale, R, T, device):  
    # 将尺度、旋转和平移转换为4x4齐次变换矩阵  
    trf = torch.eye(4, device=device)  
    trf[:3, :3] = R * scale  # 旋转部分乘以尺度  
    trf[:3, 3] = T.ravel()   # 平移部分  
    return trf

# 最小生成树初始化中的位姿恢复
# 在构建最小生成树时，系统使用刚性点配准来逐步添加新的相机：

# 对每个新边，使用点配准对齐预测点云与已有点云  
if i in done:  
    i_j = edge_str(i, j)  
    s, R, T = rigid_points_registration(pred_i[i_j], pts3d[i], conf=conf_i[i_j])  
    trf = sRT_to_4x4(s, R, T, device)  
    pts3d[j] = geotrf(trf, pred_j[i_j])  # 变换新相机的点云
```

```python
# 使用特殊字典类记录图像对（边）中第一张图像的pointmap
self.pred_i = NoGradParamDict({ij: pred1_pts[n] for n, ij in enumerate(self.str_edges)})
# 使用特殊字典类记录图像对（边）中第二张图像的pointmap
self.pred_j = NoGradParamDict({ij: pred2_pts[n] for n, ij in enumerate(self.str_edges)})

# 数据来源
pred1_pts = pred1['pts3d']           # 第一个视图的3D点云  
pred2_pts = pred2['pts3d_in_other_view']  # 第二个视图在第一个视图坐标系中的3D点云

# 边标识符生成
def str_edges(self):  
    return [edge_str(i, j) for i, j in self.edges]
def edge_str(i, j):
    return f'{i}_{j}'

s, R, T = rigid_points_registration(self.pred_i[i_j], pts3d[i], conf=self.conf_i[i_j])

'''
# 初始化最强边作为世界坐标系原点  
score, i, j = todo.pop()  
i_j = edge_str(i, j)  
pts3d[i] = pred_i[i_j].clone()  # 第一个视图作为世界坐标系原点  
pts3d[j] = pred_j[i_j].clone()  # 第二个视图在第一个视图坐标系中  
done = {i, j}
然后通过最小生成树算法逐步添加新的视图：

# 当节点i已完成时，对齐pred[i]并设置j  
s, R, T = rigid_points_registration(pred_i[i_j], pts3d[i], conf=conf_i[i_j])  
trf = sRT_to_4x4(s, R, T, device)  
pts3d[j] = geotrf(trf, pred_j[i_j])  # 将新视图变换到世界坐标系
'''

# 为每个图像对计算相机到世界的变换  
s, R, T = rigid_points_registration(self.pred_i[i_j], pts3d[i], conf=self.conf_i[i_j])  
self._set_pose(self.pw_poses, e, R, T, scale=s)

aligned_pred_i = geotrf(pw_poses[e], pw_adapt[e] * self.pred_i[i_j])  
aligned_pred_j = geotrf(pw_poses[e], pw_adapt[e] * self.pred_j[i_j])

```

更鲁棒的解决方案最终是依靠RANSAC 和 PnP实现。通过已知的 **3D–2D 对应关系**来估计相机相对于世界坐标系的位姿，其核心目标是求解使三维点经过相机位姿变换并投影到图像平面后，与对应二维像素点最一致的旋转R和平移t。具体过程是：在给定相机内参的前提下，利用若干对$ (X_i,x_i) $建立针孔投影约束，先通过最小样本解法（如 P3P）或线性方法得到位姿初值，再结合 RANSAC 剔除外点，最后以**最小化重投影误差**为目标进行非线性优化，从而得到稳定且几何一致的相机相对位姿估计。  

```python
def fast_pnp(pts3d, focal, msk, device, pp=None, niter_PnP=10):  
    """  
    使用RANSAC-PnP算法求解相机绝对位姿  
      
    Args:  
        pts3d: 3D点云 (H, W, 3)  
        focal: 相机焦距  
        msk: 有效性掩码，指示哪些点可用于PnP求解  
        device: 计算设备  
        pp: 主点坐标 (principal point)  
        niter_PnP: RANSAC迭代次数  
      
    Returns:  
        tuple: (最优焦距, 相机到世界的变换矩阵) 或 None  
    """  
    # 检查有效点数量，PnP至少需要4个点  
    if msk.sum() < 4:  
        return None  # we need at least 4 points for PnP  
    pts3d, msk = map(to_numpy, (pts3d, msk))  
  
    H, W, THREE = pts3d.shape  
    assert THREE == 3  
    # 生成像素坐标网格  
    pixels = pixel_grid(H, W)  
  
    # 如果没有提供焦距，生成多个候选焦距进行尝试  
    if focal is None:  
        S = max(W, H)  
        # 生成几何间隔的焦距候选值  
        tentative_focals = np.geomspace(S/2, S*3, 21)  
    else:  
        tentative_focals = [focal]  
  
    # 设置主点坐标，默认为图像中心  
    if pp is None:  
        pp = (W/2, H/2)  
    else:  
        pp = to_numpy(pp)  
  
    best = 0,  # 初始化最佳结果  
    # 遍历所有候选焦距  
    for focal in tentative_focals:  
        # 构建相机内参矩阵K  
        K = np.float32([(focal, 0, pp[0]), (0, focal, pp[1]), (0, 0, 1)])  
  
        # 使用OpenCV的RANSAC-PnP求解器  
        success, R, T, inliers = cv2.solvePnPRansac(  
            pts3d[msk], pixels[msk], K, None,  
            iterationsCount=niter_PnP,   
            reprojectionError=5,   
            flags=cv2.SOLVEPNP_SQPNP)  
        if not success:  
            continue  
  
        # 根据内点数量评估解的质量  
        score = len(inliers)  
        if success and score > best[0]:  
            best = score, R, T, focal  
  
    # 如果没有找到有效解，返回None  
    if not best[0]:  
        return None  
  
    # 提取最佳结果  
    _, R, T, best_focal = best  
    # 将旋转向量转换为旋转矩阵 (world to cam)  
    R = cv2.Rodrigues(R)[0]  
    R, T = map(torch.from_numpy, (R, T))  
    # 返回相机到世界的变换矩阵 (cam to world)  
    return best_focal, inv(sRT_to_4x4(1, R, T, device))
```



## 2.3.4 Absolute pose estimation
也称为视觉定位，同样可以通过几种不同的方式来实现。让$ I_Q $表示查询图像，$ I_B
 $表示具有 2D-3D 对应关系的参考图像。首先,$ I_Q $的内参可以根据$ X^{Q,Q} $进行估计。通过一对图像的pointmap可以获取$ I_Q $和 $ I_B $之间的 2D 对应关系，进而生成 $ I_Q $ 的 2D-3D 对应关系，然后通过 PnP-RANSAC 求解。

> 视觉定位：给定一张（或少量）查询图像，估计该相机在**已知世界坐标系**中的**绝对位姿**。  
>
> 求解PNP需要已知：n个2d-3d对应关系（3d点为世界系下的表示）、相机内参矩阵
>

```python
def fast_pnp(pts3d, focal, msk, device, pp=None, niter_PnP=10):  
    """  
    使用RANSAC-PnP算法求解相机绝对位姿  
      
    Args:  
        pts3d: 3D点云 (H, W, 3)  
        focal: 相机焦距  
        msk: 有效性掩码，指示哪些点可用于PnP求解  
        device: 计算设备  
        pp: 主点坐标 (principal point)  
        niter_PnP: RANSAC迭代次数  
      
    Returns:  
        tuple: (最优焦距, 相机到世界的变换矩阵) 或 None  
    """  
    # 检查有效点数量，PnP至少需要4个点  
    if msk.sum() < 4:  
        return None  # we need at least 4 points for PnP  
    pts3d, msk = map(to_numpy, (pts3d, msk))  
  
    H, W, THREE = pts3d.shape  
    assert THREE == 3  
    # 生成像素坐标网格  
    pixels = pixel_grid(H, W)  
  
    # 如果没有提供焦距，生成多个候选焦距进行尝试  
    if focal is None:  
        S = max(W, H)  
        # 生成几何间隔的焦距候选值  
        tentative_focals = np.geomspace(S/2, S*3, 21)  
    else:  
        tentative_focals = [focal]  
  
    # 设置主点坐标，默认为图像中心  
    if pp is None:  
        pp = (W/2, H/2)  
    else:  
        pp = to_numpy(pp)  
  
    best = 0,  # 初始化最佳结果  
    # 遍历所有候选焦距  
    for focal in tentative_focals:  
        # 构建相机内参矩阵K  
        K = np.float32([(focal, 0, pp[0]), (0, focal, pp[1]), (0, 0, 1)])  
  
        # 使用OpenCV的RANSAC-PnP求解器  
        success, R, T, inliers = cv2.solvePnPRansac(  
            pts3d[msk], pixels[msk], K, None,  
            iterationsCount=niter_PnP,   
            reprojectionError=5,   
            flags=cv2.SOLVEPNP_SQPNP)  
        if not success:  
            continue  
  
        # 根据内点数量评估解的质量  
        score = len(inliers)  
        if success and score > best[0]:  
            best = score, R, T, focal  
  
    # 如果没有找到有效解，返回None  
    if not best[0]:  
        return None  
  
    # 提取最佳结果  
    _, R, T, best_focal = best  
    # 将旋转向量转换为旋转矩阵 (world to cam)  
    R = cv2.Rodrigues(R)[0]  
    R, T = map(torch.from_numpy, (R, T))  
    # 返回相机到世界的变换矩阵 (cam to world)  
    return best_focal, inv(sRT_to_4x4(1, R, T, device))


def minimum_spanning_tree(imshapes, edges, pred_i, pred_j, conf_i, conf_j, im_conf, min_conf_thr,  
                          device, has_im_poses=True, niter_PnP=10, verbose=True):  
    """  
    使用最小生成树算法初始化多视图相机的位姿  
      
    Args:  
        imshapes: 每个图像的形状列表 [(H1,W1), (H2,W2), ...]  
        edges: 图像对边列表 [(i,j), (k,l), ...]  
        pred_i/pred_j: 每个边的3D点预测字典 {edge_str: pts3d}  
        conf_i/conf_j: 每个边的置信度字典 {edge_str: conf}  
        im_conf: 每个图像的置信度列表 [conf1, conf2, ...]  
        min_conf_thr: 最小置信度阈值  
        device: 计算设备  
        has_im_poses: 是否计算图像级位姿  
        niter_PnP: PnP算法迭代次数  
        verbose: 是否打印详细信息  
      
    Returns:  
        tuple: (pts3d列表, MST边列表, 焦距列表, 位姿列表)  
    """  
    n_imgs = len(imshapes)  
      
    # 第一步：构建稀疏图并计算最小生成树  
    # 根据置信度计算边权重（置信度越高，权重越小，因为后面取负号）  
    sparse_graph = -dict_to_sparse_graph(compute_edge_scores(map(i_j_ij, edges), conf_i, conf_j))  
    # 使用scipy计算最小生成树  
    msp = sp.csgraph.minimum_spanning_tree(sparse_graph).tocoo()  
  
    # 初始化数据结构  
    pts3d = [None] * len(imshapes)  # 存储每个图像的3D点云  
    todo = sorted(zip(-msp.data, msp.row, msp.col))  # 按权重排序的边列表（从强到弱）  
    im_poses = [None] * n_imgs  # 存储每个图像的相机位姿  
    im_focals = [None] * n_imgs  # 存储每个图像的焦距  
  
    # 第二步：用最强的边初始化坐标系  
    score, i, j = todo.pop()  # 取出权重最大的边  
    if verbose:  
        print(f' init edge ({i}*,{j}*) {score=}')  
    i_j = edge_str(i, j)  
    # 设置前两个相机的3D点云作为初始坐标系  
    pts3d[i] = pred_i[i_j].clone()  
    pts3d[j] = pred_j[i_j].clone()  
    done = {i, j}  # 标记已处理的相机  
      
    # 如果需要计算图像位姿，设置第一个相机为世界坐标系原点  
    if has_im_poses:  
        im_poses[i] = torch.eye(4, device=device)  # 第一个相机位姿为单位矩阵  
        im_focals[i] = estimate_focal(pred_i[i_j])  # 估计第一个相机的焦距  
  
    # 第三步：按照MST顺序逐步添加相机  
    msp_edges = [(i, j)]  # 记录MST的边  
    while todo:  
        score, i, j = todo.pop()  # 取出下一个最强的边  
  
        # 如果相机i的焦距未知，尝试估计  
        if im_focals[i] is None:  
            im_focals[i] = estimate_focal(pred_i[i_j])  
  
        # 情况1：相机i已处理，需要添加相机j  
        if i in done:  
            if verbose:  
                print(f' init edge ({i},{j}*) {score=}')  
            assert j not in done  
            i_j = edge_str(i, j)  
            # 使用刚性点配准将pred_i[i_j]对齐到已知的pts3d[i]  
            s, R, T = rigid_points_registration(pred_i[i_j], pts3d[i], conf=conf_i[i_j])  
            trf = sRT_to_4x4(s, R, T, device)  
            # 应用变换到pred_j[i_j]，得到相机j的3D点云  
            pts3d[j] = geotrf(trf, pred_j[i_j])  
            done.add(j)  
            msp_edges.append((i, j))  
  
            # 如果需要且相机i位姿未知，设置相机i的位姿  
            if has_im_poses and im_poses[i] is None:  
                im_poses[i] = sRT_to_4x4(1, R, T, device)  
  
        # 情况2：相机j已处理，需要添加相机i  
        elif j in done:  
            if verbose:  
                print(f' init edge ({i}*,{j}) {score=}')  
            assert i not in done  
            i_j = edge_str(i, j)  
            # 使用刚性点配准将pred_j[i_j]对齐到已知的pts3d[j]  
            s, R, T = rigid_points_registration(pred_j[i_j], pts3d[j], conf=conf_j[i_j])  
            trf = sRT_to_4x4(s, R, T, device)  
            # 应用变换到pred_i[i_j]，得到相机i的3D点云  
            pts3d[i] = geotrf(trf, pred_i[i_j])  
            done.add(i)  
            msp_edges.append((i, j))  
  
            # 如果需要且相机i位姿未知，设置相机i的位姿  
            if has_im_poses and im_poses[i] is None:  
                im_poses[i] = sRT_to_4x4(1, R, T, device)  
        else:  
            # 情况3：两个相机都未处理，稍后重试  
            todo.insert(0, (score, i, j))  
  
    # 第四步：使用PnP填充缺失的位姿信息  
    if has_im_poses:  
        # 按置信度从高到低排序所有边  
        pair_scores = list(sparse_graph.values())  # 已经是负分数：越小越好  
        edges_from_best_to_worse = np.array(list(sparse_graph.keys()))[np.argsort(pair_scores)]  
          
        # 首先填充缺失的焦距  
        for i, j in edges_from_best_to_worse.tolist():  
            if im_focals[i] is None:  
                im_focals[i] = estimate_focal(pred_i[edge_str(i, j)])  
  
        # 然后使用PnP求解缺失的位姿  
        for i in range(n_imgs):  
            if im_poses[i] is None:  
                # 使用高置信度点进行PnP求解  
                msk = im_conf[i] > min_conf_thr  
                res = fast_pnp(pts3d[i], im_focals[i], msk=msk, device=device, niter_PnP=niter_PnP)  
                if res:  
                    im_focals[i], im_poses[i] = res  
            # 如果PnP失败，使用单位矩阵作为默认位姿  
            if im_poses[i] is None:  
                im_poses[i] = torch.eye(4, device=device)  
          
        im_poses = torch.stack(im_poses)  # 将位姿列表堆叠成张量  
    else:  
        im_poses = im_focals = None  
  
    return pts3d, msp_edges, im_focals, im_poses


```

另一种解决方案是获取$ I^Q $和$ I^B
 $之间的相对位姿，如前所述。然后，我们根据$ X^{B,B} $和$ I^B
 $的地面实况点图之间的比例，通过适当缩放该姿势，将其转换为世界坐标。



## 2.4 Global Alignment
目前呈现的网络 F 只能处理一对图像。我们现在提出一种快速且简单的后端处理优化方法，用于整个场景，使得从多张图像预测的点图能够对齐到一个联合的 3D 空间中。这得益于Dust3R pointmap的丰富内容，这些pointmap在设计上包含了两个对齐的点云及其对应的像素到 3D 的映射。



**成对图（Pairwise graph）**

给定一组图像$ \left\{I^1,I^2,...,I^N \right\} $对于给定场景，我们首先构造一个连通图 G(V, E)，其中 N 个图像形成顶点 V，每条边 e = (n, m) ∈ E 表示图像$ I^n $和$ I^m $共享的一些视觉内容。并根据两对的平均置信度来测量它们的重叠，然后我们过滤掉低置信度对。



**全局优化（Global optimization）  **

我们使用连接图 G 来恢复所有相机 n = 1...N 的全局对齐点图。为此，我们首先对每个图像对 $ e = (n,m)\in E
 $预测成对点$ X^{n,n} $和$ X^{m,n} $及其相关的置信度图 $ C^{n,n} $, $ C^{m,n} $。为了清楚起见，让我们定义$ X^{n,e}:=X^{n,n} $ 和 $ X^{m,e}:=X^{m,n} $。由于我们的目标涉及在公共坐标系中旋转所有成对预测，因此我们引入了成对姿势 $ P_e $ 和与每对 $ e \in E
 $ 相关的缩放 $ \sigma_e >0  $。然后，我们制定以下优化问题：

$ X^* = \mathop{argmin}\limits_{x,P,\sigma}\sum_{e\in \varepsilon}\sum_{v\in e}\sum_{i=1}^{HW}C_i^{v,e}||x_i^v-\sigma_eP_eX_i^{v,e}|| $

对于给定的边 e，相同的刚性变换$ P_e $应该将点图$ X^{n,e} $和$ X^{m,e} $与世界坐标点图$ x^n $和$ x^m $对齐，因为根据定义，$ X^{n,e} $和 $ X^{m,e} $都在同一坐标系中表示。

我们指出，与传统BA相反，这种全局优化在实践中执行起来快速且简单。事实上，我们并不是像束调整通常那样最小化 2D 重投影误差，而是最小化 3D 投影误差。优化是使用标准梯度下降进行的，通常在几百步后收敛，在标准 GPU 上只需要几秒钟。

```python
# 全局优化Loss

# 创建优化器，根据模式选择优化器类型  
scene = global_aligner(output, device=device, mode=GlobalAlignerMode.Point

# 位姿初始化，使用最小生成树算法初始化相机位姿：
def compute_global_alignment(self, init=None, niter_PnP=10, **kw):  
    if init == 'mst':  
        init_fun.init_minimum_spanning_tree(self, niter_PnP=niter_PnP)  
    return global_alignment_loop(self, **kw)

# 主优化循环
global_alignment_loop 实现迭代优化过程 base_opt.py:326-349 ：

def global_alignment_loop(net, lr=0.01, niter=300, schedule='cosine', lr_min=1e-6):  
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.9))  
      
    for n in range(niter):  
        loss, _ = global_alignment_iter(net, n, niter, lr_base, lr_min, optimizer, schedule)  
    return loss

# 单步迭代，每次迭代包含前向传播、损失计算和参数更新 base_opt.py:352-366 ：
def global_alignment_iter(net, cur_iter, niter, lr_base, lr_min, optimizer, schedule):  
    lr = cosine_schedule(t, lr_base, lr_min)  # 学习率调度  
    adjust_learning_rate_by_lr(optimizer, lr)  
    optimizer.zero_grad()  
    loss = net()  # 前向传播  
    loss.backward()  # 反向传播  
    optimizer.step()  # 参数更新  
    return float(loss), lr

# 损失计算前向传播，在 PointCloudOptimizer.forward() 中计算对齐损失：

def forward(self):  
    pw_poses = self.get_pw_poses()  # 获取成对位姿  
    pw_adapt = self.get_adaptors().unsqueeze(1)  
    proj_pts3d = self.get_pts3d(raw=True)  # 获取世界坐标系3D点  
      
    # 根据位姿变换预测点云  
    aligned_pred_i = geotrf(pw_poses, pw_adapt * self._stacked_pred_i)  
    aligned_pred_j = geotrf(pw_poses, pw_adapt * self._stacked_pred_j)  
      
    # 计算距离损失  
    li = self.dist(proj_pts3d[self._ei], aligned_pred_i, weight=self._weight_i).sum() / self.total_area_i  
    lj = self.dist(proj_pts3d[self._ej], aligned_pred_j, weight=self._weight_j).sum() / self.total_area_j  
    return li + lj
    
'''
参数管理
优化的参数类型
PointCloudOptimizer 管理以下可优化参数 

im_depthmaps: 深度图（对数空间）
im_poses: 相机位姿（四元数+平移）
im_focals: 相机焦距（对数空间）
im_pp: 主点坐标
参数获取方法
提供多种方法获取优化后的参数 optimizer.py:152-186 ：

def get_im_poses(self):  # 相机位姿  
def get_focals(self):    # 焦距  
def get_depthmaps(self): # 深度图  
def get_pts3d(self):     # 3D点云
'''

# 后处理-点云清理，clean_pointcloud 函数移除不一致的点：

def clean_pointcloud(im_confs, K, cams, depthmaps, all_pts3d, tol=0.001, bad_conf=0):  
    # 将所有3D点投影到每个相机坐标系  
    # 如果点在深度图前面，降低其置信度

-------------------------------------------------------------------------------------
# 全局对齐  
    
# 最小生成树构建
'''
minimum_spanning_tree 函数首先构建基于置信度的稀疏图，
然后计算最小生成树
'''
def minimum_spanning_tree(imshapes, edges, pred_i, pred_j, conf_i, conf_j, im_conf, min_conf_thr,  
                          device, has_im_poses=True, niter_PnP=10, verbose=True):  
    n_imgs = len(imshapes)  
    sparse_graph = -dict_to_sparse_graph(compute_edge_scores(map(i_j_ij, edges), conf_i, conf_j))  
    msp = sp.csgraph.minimum_spanning_tree(sparse_graph).tocoo()

# 世界坐标系原点设置,算法选择置信度最高的边作为世界坐标系原点
# init with strongest edge  
score, i, j = todo.pop()  
i_j = edge_str(i, j)  
pts3d[i] = pred_i[i_j].clone()  # 第一个视图作为世界坐标系原点  
pts3d[j] = pred_j[i_j].clone()  # 第二个视图在第一个视图坐标系中  
done = {i, j}  
if has_im_poses:  
    im_poses[i] = torch.eye(4, device=device)  # 第一个相机位姿设为单位矩阵

# 增量式对齐新视图,通过最小生成树逐步添加新视图，使用刚体点配准对齐到世界坐标系 ：
if i in done:  
    # align pred[i] with pts3d[i], and then set j accordingly  
    i_j = edge_str(i, j)  
    s, R, T = rigid_points_registration(pred_i[i_j], pts3d[i], conf=conf_i[i_j])  
    trf = sRT_to_4x4(s, R, T, device)  
    pts3d[j] = geotrf(trf, pred_j[i_j])  # 将新视图变换到世界坐标系

# 初始化的位姿类型
# 1. 成对位姿（Pairwise Poses）,在 init_from_pts3d 中为每个图像对设置相机到世界的变换

# set all pairwise poses  
for e, (i, j) in enumerate(self.edges):  
    i_j = edge_str(i, j)  
    # compute transform that goes from cam to world  
    s, R, T = rigid_points_registration(self.pred_i[i_j], pts3d[i], conf=self.conf_i[i_j])  
    self._set_pose(self.pw_poses, e, R, T, scale=s)

#2. 图像级位姿（Image-level Poses）,为每个图像设置独立的相机位姿
# init all image poses  
if self.has_im_poses:  
    for i in range(self.n_imgs):  
        cam2world = im_poses[i]  
        depth = geotrf(inv(cam2world), pts3d[i])[..., 2]  
        self._set_depthmap(i, depth)  
        self._set_pose(self.im_poses, i, cam2world)
'''
图片对齐到世界坐标系的过程
1. 坐标系传递机制
第一步: 选择最强边 (i,j)，将图像 i 设为世界坐标系原点
第二步: 图像 j 的点云直接使用 pred_j[i_j]（已在 i 的坐标系中）
第三步: 对于新边 (i,k)，通过刚体配准计算变换，将 k 对齐到世界坐标系
2. 刚体变换计算
使用 rigid_points_registration 计算从相机坐标系到世界坐标系的变换 init_im_poses.py:220-223 ：

def rigid_points_registration(pts1, pts2, conf):  
    R, T, s = roma.rigid_points_registration(  
        pts1.reshape(-1, 3), pts2.reshape(-1, 3), weights=conf.ravel(), compute_scaling=True)  
    return s, R, T
3. 变换应用
通过 geotrf 函数应用变换将新视图的点云对齐到世界坐标系 init_im_poses.py:164-165 ：

pts3d[j] = geotrf(trf, pred_j[i_j])
'''

                       
```



**恢复相机参数（Recovering camera parameters）**

对全局优化框架的直接扩展可以恢复所有相机参数。通过简单地替换

$ x_{i,j}^n := P_n^{-1}h(K_n^{-1}[iD_{i,j}^n;jD_{i,j}^n;D_{i,j}^n]) $（即强制标准相机针孔模型）

我们可以估计所有相机位姿 {Pn}、关联的内在函数 {Kn} 和深度图 {Dn} (n=1...N)

```python
# 相机参数初始化
def __init__(self, *args, optimize_pp=False, focal_break=20, **kwargs):  
    super().__init__(*args, **kwargs)  
  
    self.has_im_poses = True  # 标记此类支持图像级位姿优化  
    self.focal_break = focal_break  # 焦距缩放因子  
  
    # 初始化可优化参数  
    # 深度图（对数空间）  
    self.im_depthmaps = nn.ParameterList(torch.randn(H, W)/10-3 for H, W in self.imshapes)  
    # 相机位姿（7维：四元数旋转+3D平移）  
    self.im_poses = nn.ParameterList(self.rand_pose(self.POSE_DIM) for _ in range(self.n_imgs))  
    # 相机焦距（对数空间）  
    self.im_focals = nn.ParameterList(torch.FloatTensor(  
        [self.focal_break*np.log(max(H, W))]) for H, W in self.imshapes)  
    # 主点坐标  
    self.im_pp = nn.ParameterList(torch.zeros((2,)) for _ in range(self.n_imgs))  
    self.im_pp.requires_grad_(optimize_pp)  # 控制是否优化主点



#设置已知焦距
def preset_focal(self, known_focals, msk=None):  
    """  
    设置已知的相机焦距  
      
    Args:  
        known_focals: 已知焦距列表  
        msk: 焦距掩码，指定哪些相机需要设置  
    """  
    self._check_all_imgs_are_selected(msk)  
  
    for idx, focal in zip(self._get_msk_indices(msk), known_focals):  
        if self.verbose:  
            print(f' (setting focal #{idx} = {focal})')  
        self._no_grad(self._set_focal(idx, focal))  
  
    self.im_focals.requires_grad_(False)  # 固定焦距，不参与优化

# 内部焦距设置方法
def _set_focal(self, idx, focal, force=False):  
    """  
    内部方法：设置单个相机的焦距（在对数空间）  
      
    Args:  
        idx: 相机索引  
        focal: 焦距值  
        force: 是否强制设置  
    """  
    param = self.im_focals[idx]  
    if param.requires_grad or force:  
        # 将焦距转换到对数空间并存储  
        param.data[:] = self.focal_break * np.log(focal)  
    return param

# 获取优化后的焦距
def get_focals(self):  
    """  
    获取优化后的相机焦距（从对数空间转换回线性空间）  
      
    Returns:  
        torch.Tensor: (n_imgs,) 焦距值  
    """  
    log_focals = torch.stack(list(self.im_focals), dim=0)  
    return (log_focals / self.focal_break).exp()

#主点参数设置与获取
# 设置已知主点
def preset_principal_point(self, known_pp, msk=None):  
    """  
    设置已知的相机主点坐标  
      
    Args:  
        known_pp: 已知主点坐标列表  
        msk: 主点掩码，指定哪些相机需要设置  
    """  
    self._check_all_imgs_are_selected(msk)  
  
    for idx, pp in zip(self._get_msk_indices(msk), known_pp):  
        if self.verbose:  
            print(f' (setting principal point #{idx} = {pp})')  
        self._no_grad(self._set_principal_point(idx, pp))  
  
    self.im_pp.requires_grad_(False)  # 固定主点，不参与优化
# 获取优化后的主点
def get_principal_points(self):  
    """  
    获取优化后的相机主点坐标  
      
    Returns:  
        torch.Tensor: (n_imgs, 2) 主点坐标  
    """  
    return self._pp + 10 * self.im_pp  # 从偏移量恢复实际坐标
    
# 深度图参数设置与获取
# 设置深度图
def _set_depthmap(self, idx, depth, force=False):  
    """  
    设置单个相机的深度图（在对数空间）  
      
    Args:  
        idx: 相机索引  
        depth: 深度图  
        force: 是否强制设置  
    """  
    depth = _ravel_hw(depth, self.max_area)  # 展平深度图  
  
    param = self.im_depthmaps[idx]  
    if param.requires_grad or force:  
        # 将深度转换到对数空间并处理无效值  
        param.data[:] = depth.log().nan_to_num(neginf=0)  
    return param
# 获取优化后的深度图
def get_depthmaps(self, raw=False):  
    """  
    获取优化后的深度图  
      
    Args:  
        raw: 是否返回展平的原始格式  
      
    Returns:  
        list: 深度图列表  
    """  
    res = self.im_depthmaps.exp()  # 从对数空间转换回线性空间  
    if not raw:  
        # 恢复原始图像尺寸  
        res = [dm[:h*w].view(h, w) for dm, (h, w) in zip(res, self.imshapes)]  
    return res
# 完整的内参矩阵获取
def get_intrinsics(self):  
    """  
    获取完整的相机内参矩阵  
      
    Returns:  
        torch.Tensor: (n_imgs, 3, 3) 内参矩阵  
    """  
    K = torch.zeros((self.n_imgs, 3, 3), device=self.device)  
    focals = self.get_focals().flatten()  
    K[:, 0, 0] = K[:, 1, 1] = focals  # 设置焦距  
    K[:, :2, 2] = self.get_principal_points()  # 设置主点  
    K[:, 2, 2] = 1  
    return K


```



<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2025/png/58377837/1766302622545-8b0a045c-e6e2-4511-ae67-b94af1dd1310.png)





# 3 experiments
**训练数据**

我们使用八个数据集的混合来训练我们的网络：**Habitat 、MegaDepth 、ARKitScenes 、Static Scenes 3D 、Blended MVS 、ScanNet++ 、CO3D-v2 和 Waymo 。**

这些数据集具有不同的场景类型：室内、室外、合成、真实世界、以对象为中心等。当数据集没有直接提供图像对时，我们根据Crocov2中描述的方法提取它们。具体来说，利用现成的图像检索和点匹配算法来匹配和验证图像对，总共提取了 850 万对。



**训练细节**

在每一个epoch,**从每个数据集随机采样相同数量图像对**，以均衡数据集大小差异。我们希望向网络提供相对高分辨率的图像，例如最大尺寸为 512 像素。首先在 224×224 图像上训练，然后在更大的 512 像素图像上训练。我们**为每个批次随机选择图像长宽比（例如 16/9、4/3 等）**，以便在测试时我们的网络熟悉不同的图像形状。我们只需将图像裁剪为所需的纵横比，然后调整大小以使最大尺寸为 512 像素。**在训练之前，我们使用现成的 CroCo 预训练模型的权重来初始化我们的网络。**

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2025/png/58377837/1766292227970-597bcfc9-275f-43cd-9f1c-0fee88baae6c.png)



## 3.1 Visual Localization
首先评估 DUSt3R 在 7Scenes 和 Cambridge Landmarks 数据集上的绝对姿态估计任务，**分别以 (cm/°) 为单位报告平移误差和旋转误差中值。 **

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2025/png/58377837/1766232982317-f7563a30-74b5-41a0-87eb-b73a771b4fff.png)

我们将两个数据集的每个场景的结果与表 1 中的最新技术进行比较。与现有方法（特征匹配方法或基于端到端学习的方法）相比，**我们的方法获得了相当的准确性**，甚至在某些情况下能够超越像 Hloc 这样的强基线。**我们认为这很重要，因为DUSt3R 从未接受过任何视觉定位训练；其次，在 DUSt3R 的训练过程中，既没有看到查询图像，也没有看到数据库图像。**



## 3.2 Multi-view Pose Estimation
我们使用两个多视图数据集CO3Dv2和RealEstate10k进行评估。我们将从 PnP-RANSAC 或全局对齐获得的 DUSt3R 姿态估计结果与基于学习的 RelPose、PoseReg和 PoseDiffusion 以及基于结构的 PixSFM、COLMAP+SPSG（COLMAP 用 SuperPoint 和 SuperGlue 扩展）进行比较。我们**使用图像对的相对旋转精度（RRA）和相对平移精度（RTA）来评估相对位姿误差，并选择阈值τ = 15来报告RTA@15和RRA@15。**此外，我们还计算了平均精度 (mAA)@30，定义为 min(RRA@30、RTA@30) 处角度差精度曲线下的面积。

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2025/png/58377837/1766234260454-efa8544f-a26c-47fa-ba6a-68f729d67a63.png)

**具有全局对齐的 DUSt3R 在两个数据集上实现了最佳的整体性能，并显着超越了最先进的 PoseDiffusion 。此外，具有 PnP 的 DUSt3R 还表现出优于学习和基于结构的现有方法的性能。**

## 3.3 Monocular Depth
对于这个单目任务，**只需将相同的输入图像 I 作为 F(I,I) 提供给网络。根据设计，深度预测只是预测 3D 点图中的 z 坐标。**

我们在两个室外（DDAD 、KITTI ）和三个室内（NYUv2 、BONN 、TUM ）数据集上对 DUSt3R 进行基准测试。我们将 DUSt3R 的性能与**分类为监督、自监督和零样本设置**的最先进方法进行比较，最后一个类别对应于 DUSt3R。我们使用单目深度评估中常用的两个指标：**目标 **$ y $** 和预测 **$ \hat y
 $** 之间的绝对相对误差 AbsRel，**

$ AbsRel = |y-\hat y|/y $**，以及预测阈值精度**$ \delta_{1.25} = max(\hat y/y,y/\hat y) < 1.25 $**.**

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2025/png/58377837/1766234425772-cce49844-1a32-4b58-a8b3-92cf7d1a9cbc.png)

在零样本设置中，最先进的技术以最近的 SlowTv 为代表。这种方法收集了大量包含城市、自然、合成和室内场景的精选数据集，并训练了一个通用模型。对于混合中的每个数据集，相机参数是已知的或使用 COLMAP 进行估计。**如上表所示，DUSt3R 能够很好地适应室外和室内环境。它的性能优于自监督基线 Monodepth2、SC-SfM-Learners、SC-DepthV3，并且与最先进的监督基线DPT-BEiT、NeWCRFs相当。**



## 3.4 Multi-view Depth
同样，我们提取深度图作为预测点图的 z 坐标。在同一图像有多个深度图可用的情况下，我们重新调整所有预测以将它们对齐在一起，并通过按置信度加权的简单平均来聚合所有预测。数据集和指标。

我们在 DTU、ETH3D、Tanks and Temples 和 ScanNet 数据集上对其进行评估。我们测试**每个测试集上阈值为 1.03 的绝对相对误差 (rel) 和内点比率 (τ ) 以及所有测试集的平均值。**注意，我们没有利用GT相机参数和姿势，也没有利用GT深度范围，因此我们的预测仅在比例因子范围内有效。为了进行定量测量，我们使用预测深度和真实深度的中值对预测进行归一化。

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2025/png/58377837/1766284781051-076c8452-f9c9-4f1e-a7a4-31e2522d6d80.png)

我们在表 3 中观察到，**DUSt3R 在 ETH-3D 上实现了最先进的精度，并且总体上优于最新最先进的方法，即使是那些使用真实相机姿势的方法**。从时间上看，我们的方法也比传统的 COLMAP 管道快得多。这展示了我们的方法在各种领域（室内、室外、小规模或大规模场景）的适用性，同时没有在测试领域进行训练（除了 ScanNet 测试集），因为训练分割是 Habitat 数据集的一部分。



## 3.5 3D Reconstruction
最后，测量在第2.4节中描述的全局对齐过程后获得的完整重建的质量。**我们再次强调，我们的方法是第一个实现全局无约束 MVS 的方法，因为我们没有关于相机内在和外在参数的先验知识。**为了量化重建的质量，我们只需将预测与地面GT坐标系对齐。这是通过将参数固定为2.4节中的常量来完成的。这致使在地面实况坐标系中表达一致的 3D 重建。

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2025/png/58377837/1766285790664-47e929dd-af7a-474d-99f8-33577adced3e.png)

我们评估我们对 DTU 数据集的预测。我们在零样本设置中应用我们的网络，即我们不对 DTU 训练集进行微调并按原样应用我们的模型。在表4我们报告了基准作者提供的**平均准确性、平均完整性和总体平均误差指标**。重建形状的点的准确性被定义为到GT值的最小欧几里德距离，并且地面实况的点的完整性被定义为到重建形状的最小欧几里德距离。总体只是之前两个指标的平均值。

**我们的方法没有达到最佳方法的准确度水平。这些方法都利用了GT姿态**，并在适用时专门在DTU训练集上进行训练。此外，该任务的最佳结果通常是通过亚像素精度的三角测量获得的，这需要使用明确的相机参数，**而我们的方法依赖于回归，而回归的准确性通常较低。**然而，在不了解相机的先验情况下，我们仍然达到了平均精度为2.7毫米，完整性为0.8毫米，总体平均距离为1.7毫米。我们认为，这种精度水平在实际应用中非常有用，尤其考虑到我们方法的即插即用特性。



## 3.6 Ablations
在表1、表2、表3中，我们消融了 CroCo 预训练和图像分辨率对 DUSt3R 性能的影响。总体来看，观察到的一致性提升表明，预训练和高分辨率在现代数据驱动方法中起着关键作用。



















