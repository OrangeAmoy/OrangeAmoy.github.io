---
title: 目标跟踪KCF算法简介与优化
date: 2019-04-05 18:00:00
tags:
mathjax: true
categories:
- 技术文档
- 机器学习
---

# 1. 算法简介  
&emsp;&emsp;首先目标跟踪大致可分为单目标跟踪与多目标跟踪，本文重点描述单目标跟踪。目标跟踪解决的问题是：第一帧给出目标矩形框，然后从后续帧开始目标跟踪算法能够跟踪该目标矩形框。通常来说，目标跟踪有几大难点：外观变形，光照变化，快速运动和运动模糊，背景相似干扰：  

<div align=center>
<img src=https://s2.ax1x.com/2019/04/05/ARj13t.png width=500/>
</div>  

<!-- more -->
平面外旋转，平面内旋转，尺度变化，遮挡和出视野等情况：

<div align=center>
<img src=https://s2.ax1x.com/2019/04/05/ARXLpq.png width=500/>
</div>  

&emsp;&emsp;对于目标跟踪方法的分类，大致可分为生成模型方法和判别模型方法，目前比较流行的是判别类方法。生成类方法为在当前帧对目标区域建模，下一帧寻找与模型最相似的区域就是预测位置，例如：卡尔曼滤波，粒子滤波，mean-shift等。判别类算法的经典套路为图像特征+机器学习，当前帧以目标区域为正样本，背景区域为负样本，机器学习训练分类器，下一帧用训练好的分类器找最优区域，例如：Struck，TLD等。与生成类方法最大的区别，是分类器训练过程中用到了背景信息，这样分类器专注区分前景和背景，判别类方法普遍都比生成类好。判别类方法的最新发展就是相关滤波类方法和深度学习类方法。相关滤波方法例如：DCF，KCF，ECO等。深度学习方法例如：MDNet，TCNN，SiamFC等  
&emsp;&emsp;参考资料：  
&emsp;&emsp;[大话目标跟踪—背景介绍](https://zhuanlan.zhihu.com/p/26415747)  
&emsp;&emsp;[benchmark_results](https://github.com/foolwood/benchmark_results)

# 2. 算法案例  
## 2.1 背景介绍  
&emsp;&emsp;2017年的时候，由于项目需求，需要研究一款在速度与性能均有较好表现的单目标跟踪算法，在当时通过多种算法选型，最终选择了KCF作为主要研究对象，现主要对相关滤波算法KCF进行简单介绍。  
&emsp;&emsp;KCF是一种判别式跟踪方法，这类方法一般都是在跟踪的过程中训练一个目标检测器，使用目标检测器去检测下一帧预测位置是否是目标，然后再使用新检测结果去更新训练集进而更新目标检测器。而在训练目标检测器时一般选取目标区域为正样本，目标周围区域为负样本。  
&emsp;&emsp;KCF的主要贡献可概括为：  
- 使用目标周围区域的循环矩阵采集正负样本，利用岭回归训练目标检测器，并成功的利用循环矩阵在傅里叶空间可对角化的性质将矩阵的运算转化为向量的Hadamad积，即元素的点乘，大大降低了运算量，提高了运算速度，使算法满足实时性要求。  
- 将线性空间的岭回归通过核函数映射到非线性空间，在非线性空间通过求解一个对偶问题和某些常见的约束，同样的可以使用循环矩阵傅里叶空间对角化简化计算。
- 给出了一种将多通道数据融入该算法的途径。

## 2.2 公式推导  
### 2.2.1 一维岭回归  
#### 2.2.1.1 岭回归  
&emsp;&emsp;设训练样本集$(x_i,y_i)$，那么其线性回归函数$f(x_i)=w^Tx_i$，$w$是列向量表示权重系数，可通过最小二乘法求解  
$$\min_w\sum_i(f(x_i)-y_i)^2+\lambda\Vert w\Vert^2$$  
&emsp;&emsp;其中$\lambda$用于控制系统的结构复杂性，保证分类器的泛化性能。  
&emsp;&emsp;写成矩阵的形式  
$$\min_w\Vert Xw-y\Vert^2+\lambda\Vert w\Vert^2$$  
&emsp;&emsp;其中$X=[x_1,x_2,\cdots,x_n]^T$的每一行表示一个向量，$y$是列向量，每个元素对应一个样本的标签，于是令导数为0，可求得  
$$w=(X^TX+\lambda I)^{-1}X^Ty$$  
&emsp;&emsp;写成复数域中形式  
$$w=(X^HX+\lambda I)^{-1}X^Hy$$  
&emsp;&emsp;其中$X^H$表示复共轭转置矩阵，$X^H=(x^*)^T$。  
#### 2.2.1.2 循环矩阵  
&emsp;&emsp;KCF中所有的训练样本是由目标样本循环位移得到的，向量的循环可由排列矩阵得到，比如  
$$x=[x_1,x_2,\cdots,x_n]^T$$  
$$P=
    \left[
    \begin{matrix}
    0 & 0 & \cdots & 0 & 1 \\
    1 & 0 & \cdots & 0 & 0 \\
    0 & 1 & \cdots & 0 & 0 \\
    \vdots & \vdots  & \ddots & \vdots & \vdots \\
    0 & 0 & \cdots & 1 & 0 
    \end{matrix} 
    \right]$$  
$$Px=[x_n,x_1,x_2,\cdots,x_{n-1}]^T$$  
&emsp;&emsp;对于二维图像的话，可以通过$x$轴和$y$轴分别循环移动实现不同位置的移动。  
&emsp;&emsp;所以由一个向量$x\in \Re^n$可以通过不断的乘上排列矩阵得到$n$个循环移位向量,将这$n$个向量依序排列到一个矩阵中，就形成了$x$生成的循环矩阵，表示成$C(x)$  

<div align=center>
<img src=https://s2.ax1x.com/2019/04/05/ARX5nS.png width=400/>
</div>
<div align=center>
<img src=https://s2.ax1x.com/2019/04/05/ARXI0g.png width=400/>
</div>  

#### 2.2.1.3 循环矩阵傅氏空间对角化  
&emsp;&emsp;所有的循环矩阵都能够在傅氏空间中使用离散傅里叶矩阵进行对角化  
$$X=Fdiag(\hat{x})F^H$$  
&emsp;&emsp;其中$\hat{x}$对应于生成$X$的向量$x$(就是$X$的第一行矩阵)的傅里叶变化后的值，$\hat{x}=\mathscr{F}(x)=\sqrt nFx$，$F$是离散傅里叶矩阵，是常量  
$$F=\frac{1}{\sqrt{n}}
\left[
\begin{matrix}
1 & 1 & \cdots & 1 & 1 \\
1 & w & \cdots & w^{n-2} & w^{n-1} \\
1 & w^2 & \cdots & w^{2(n-2)} & w^{2(n-1)} \\
\vdots & \vdots & \ddots & \vdots & \vdots \\
1 & w^{n-1} & \cdots & w^{(n-1)(n-2)} & w^{(n-1)^2} 
\end{matrix}
\right]$$  
&emsp;&emsp;关于矩阵的傅里叶对角化可参考[循环矩阵傅里叶对角化](https://blog.csdn.net/shenxiaolu1984/article/details/50884830)。  
#### 2.2.1.4 傅氏对角化简化的岭回归  
&emsp;&emsp;将$X=Fdiag(\hat{x})F^H$带入岭回归公式得到  
$$\begin{aligned}
w & = (Fdiag(\hat{x}^*)F^HFdiag(\hat{x})F^H+\lambda F^H)^{-1}Fdiag(\hat{x}^*)F^Hy \\
    & = (Fdiag(\hat{x}^*\odot\hat{x}+\lambda)F^H)^{-1}Fdiag(\hat{x}^*)F^Hy \\
    & = Fdiag(\frac{\hat{x}^*}{\hat{x}^*\odot\hat{x}+\lambda})F^Hy
\end{aligned}$$  
&emsp;&emsp;注意这里的分号是点除运算，就是对应元素相除。因为$\mathscr{F}(C(x)y)=\mathscr{F}^*(x)\odot\mathscr{F}(y)$，(循环矩阵傅里叶对角化)对上式两边同时傅氏变换得  
$$\mathscr{F}(w)=\mathscr{F}^*(\mathscr{F}^{-1}(\frac{\hat{x}^*}{\hat{x}^*\odot\hat{x}+\lambda}))\odot \mathscr{F}(y)$$  
&emsp;&emsp;于是
$$\hat{w}=\frac{\hat{x}\odot\hat{y}}{\hat{x}^*\odot\hat{x}+\lambda}$$  
&emsp;&emsp;这样就可以使用向量的点积运算取代矩阵运算，特别是求逆运算，大大提高了计算速度。  
$$w=\mathscr{F}^{-1}(\hat{w})$$
### 2.2.2 核空间的岭回归  
&emsp;&emsp;我们希望找到一个非线性映射函数$\phi(x)$列向量，使映射后的样本在新空间中线性可分，那么在新空间中就可以使用岭回归来寻找一个分类器$f(x_i)=w^T\phi(x_i)$，所以这时候得到的权重系数为  
$$w=\min_w\Vert \phi(X)w-y\Vert^2+\lambda\Vert w\Vert^2$$  
&emsp;&emsp;$w$是$\phi(X)=[\phi(x_1),\phi(x_2),\cdots,\phi(x_n)]^T$行向量张成的空间中的一个向量，所以令$w=\sum_i\alpha_i\phi(x_i)$上式就变为  
$$\alpha=\min_\alpha \Vert \phi(X)\phi(X)^T\alpha-y\Vert^2+\lambda\Vert \phi(X)^T\alpha\Vert^2$$  
&emsp;&emsp;该问题称为$w$的对偶问题。  
&emsp;&emsp;令关于列向量$\alpha$导数为0，  
$$\begin{aligned}
J(\alpha) & = \alpha^T\phi(X)\phi(X)^T\phi(X)\phi(X)^T\alpha-2y^T\phi(X)\phi(X)^T\alpha+C_{onstant}+\lambda\alpha^T\phi(X)\phi(X)^T\alpha \\
\frac{\partial J}{\partial\alpha}  & = 2\phi(X)\phi(X)^T\phi(X)\phi(X)^T\alpha+2\lambda\phi(X)\phi(X)^T\alpha-2\phi(X)\phi(X)^Ty=0 \\
\alpha^*  & = (\phi(X)\phi(X)^T+\lambda I)^{-1}y
\end{aligned}$$  
&emsp;&emsp;注：$\phi(X)\phi(X)^T$类似于核空间变量的协方差矩阵，矩阵的转置乘以矩阵，一定可逆。  
&emsp;&emsp;对于核方法，我们一般不知道非线性映射函数$\phi(x)$的具体形式，而只是刻画在核空间的核矩阵$\phi(X)\phi(X)^T$，那么我们令$K$表示核空间的核矩阵，由核函数得到，那么$K=\phi(X)\phi(X)^T$，于是  
$$\alpha^*=(K+\lambda I)^{-1}y$$  
$$f(z)=w^T\phi(z)=\alpha^T\phi(X)\phi(z)$$  
&emsp;&emsp;这里通过循环矩阵的傅氏对角化简化计算，所以如果希望计算$\alpha$时可以同样将矩阵求逆运算变为元素运算，就希望将$K$对角化，所以希望找到一个核函数使对应的核矩阵式循环矩阵。  
定理1  
&emsp;&emsp;即核矩阵是循环矩阵应该满足两个条件：第一个样本和第二个样本都是由生成样本循环移位产生的，可以不是由同一个样本生成；满足$K(x,x')=K(Mx,Mx')$，其中$M$是排列矩阵。  
&emsp;&emsp;证明：  
&emsp;&emsp;设$x\in \Re^n$，则$x'=P^ix,\{\forall x'\in C(x)\}$，于是  
$$\begin{aligned}
K_{ij} & = \phi(x_i)^T\phi(x_j) \\
       & = K(x_i,x_j) \\
       & = K(P^ix,P^jx) \\
       & = K(x,P^{j-i}x) \\
       & = \phi(x)^T\phi(P^{j-i}x)
\end{aligned}$$  
&emsp;&emsp;因为$K$的第一行为$\phi(x)^T\phi(x_j),\{ j=1,2,\cdots,n\}$，所以$K_{ij}$相当于将第一行的第$j-i$个元素放到$K$的第$i$行$j$列上，那么$i,j\in \{ 1,2,\cdots,n\}$就得到了循环矩阵，所以$K$是循环矩阵。证明$j-i$表示除$n$的余数，因为这个过程是循环的。  

&emsp;&emsp;证毕。  
&emsp;&emsp;若$K$是循环矩阵，则  
$$\alpha = Fdiag(\hat{K}^{xx}+\lambda)^{-1}F^Hy$$  
$$\hat{\alpha}=\frac{\hat{y}}{(\hat{K}^{xx}+\lambda)^*}$$  
&emsp;&emsp;其中$K^{xx}=\phi(x)^T\phi(X)^T$是$K$中第一行。
### 2.2.3 快速检测  
&emsp;&emsp;首先由训练样本和标签训练检测器，其中训练集是由目标区域和由其移位得到的若干样本组成，对应的标签是根据距离越近正样本可能性越大的准则赋值的，然后可以得到$\alpha$  
&emsp;&emsp;待分类样本集，即待检测样本集，是由预测区域和由其移位得到的样本集合$z_j=P^jz$，那么就可以选择$f(z_j)=\alpha^T\phi(X)\phi(z_j)$最大的样本作为检测出的新目标区域,由$z_j$判断目标移动的位置。  
&emsp;&emsp;定义$K^z$是测试样本和训练样本间在核空间的核矩阵$K_{ij}^z=\phi(z_i)^T\phi(x_j),K^z=\phi(X)\phi(Z)^T$。  
&emsp;&emsp;由于核矩阵满足$K(x,x')=K(Px,Px')$,即$K(x_i,z_j)=K(P^ix,P^jz)$类似于定理1的证明可得$K^z$是循环矩阵。  
&emsp;&emsp;于是得到各个测试样本的响应  
$$\begin{aligned}
f(z) & = (\alpha^T\phi(X)\phi(Z)^T)^T=(K^z)^T\alpha=\mathscr{F}^{-1}(\hat{f}) \\
f(z) & = Fdiag(\hat{K}^{xz})F^H\alpha \\
\hat{f} & = (\hat{K}^{xz})^*\hat{\alpha}
\end{aligned}$$  
### 2.2.4 核矩阵的快速计算  
&emsp;&emsp;现在还存在的矩阵运算就是核矩阵的第一行的计算  
$$K^{xx}=\phi(x)^T\phi(X)^T,K^{xz}=\phi(X)\phi(z)$$  
&emsp;&emsp;内积和多项式核  
&emsp;&emsp;这种核函数核矩阵可以表示成$K(x_i,x'_j)=g(x_i^T,x'_j)$，于是  
$$\begin{aligned}
K^{xx'} & = g(C(x)x')^T \\
        & = g(\mathscr{F}^{-1}(\hat{x}^*\odot\hat{x}'))^T
\end{aligned}$$  
&emsp;&emsp;因此对于多项式核$K(x_i,x_j)=(x_i^Tx_j+\alpha)^b$有  
$$K^{xx'}=((\mathscr{F}^{-1}(\hat{x}^*\odot\hat{x}'))^T+\alpha)^b$$  
&emsp;&emsp;径向基核函数  
&emsp;&emsp;比如高斯核，这类函数是$\Vert x_i-x_j \Vert^2$的函数  
$$\Vert x_i-x_j \Vert^2=(x_i-x_j)^T(x_i-x_j)=\Vert x_i \Vert^2+\Vert x_j \Vert^2-2x_i^Tx_j$$  
&emsp;&emsp;所以  
$$\begin{aligned}
K^{xx'} & = h(\Vert x \Vert^2+\Vert x' \Vert^2-2C(x)x')^T  \\
        & = h(\Vert x \Vert^2+\Vert x' \Vert^2-2\mathscr{F}^{-1}(\hat{x}^*\odot\hat{x}'))^T
\end{aligned}$$  
&emsp;&emsp;对于高斯核则有  
$$K^{xx'}=exp(-\frac{1}{\sigma}(\Vert x \Vert^2+\Vert x' \Vert^2-2\mathscr{F}^{-1}(\hat{x}^*\odot\hat{x}'))^T))$$  
&emsp;&emsp;以上就是KCF的主要推导公式，且在一维情况下推导的结果，二维情况下的推导类似，推导过程略，详细推导过程见下方参考文档：   
&emsp;&emsp;[KCF目标跟踪方法分析与总结](https://www.cnblogs.com/YiXiaoZhou/p/5925019.html)  
## 2.3 代码实现  
伪代码实现：  

<div align=center>
<img src=https://s2.ax1x.com/2019/04/05/ARXo7Q.png width=600/>
</div>  

c++代码实现：  
```c++
void KCFTracker::init(const cv::Rect &roi, cv::Mat image)
{
    _tmpl = getFeatures(image, 1);
    _prob = createGaussianPeak(size_patch[0], size_patch[1]);
    train(_tmpl, 1.0); 
}

void KCFTracker::train(cv::Mat x, float train_interp_factor)
{
    cv::Mat k = gaussianCorrelation(x, x);
    cv::Mat alphaf = complexDivision(_prob, (fftd(k) + lambda));
    
    _tmpl = (1 - train_interp_factor) * _tmpl + (train_interp_factor) * x;
    _alphaf = (1 - train_interp_factor) * _alphaf + (train_interp_factor) * alphaf;
}

cv::Point2f KCFTracker::detect(cv::Mat z, cv::Mat x, float &peak_value)
{
    cv::Mat k = gaussianCorrelation(x, z);
    cv::Mat res = (real(fftd(complexMultiplication(_alphaf, fftd(k)), true)));
    minMaxLoc(res, NULL, &pv, NULL, &pi);
    peak_value = (float) pv;
}

cv::Rect KCFTracker::update(cv::Mat image)
{
    cv::Point2f res = detect(_tmpl, getFeatures(image, 0, 1.0f), peak_value);

    if (scale_step != 1) {
        cv::Point2f new_res_smaller = detect(.., getFeatures(.., 1.0f / scale_step),..);
        cv::Point2f new_res_bigger = detect(.., getFeatures(.., scale_step),..);
        ……
    }

    _roi.x = cx - _roi.width / 2.0f + ((float) res.x * cell_size * _scale);
    _roi.y = cy - _roi.height / 2.0f + ((float) res.y * cell_size * _scale);

    cv::Mat x = getFeatures(image, 0);
    train(x, interp_factor);

    return _roi;
}
```

## 2.4 KCF可视化  

<div align=center>
<img src=https://s2.ax1x.com/2019/04/05/ARXHts.png />
</div>  

# 3. 算法优化  
&emsp;&emsp;在实际开发的过程中，从刚开始接触KCF源码，到算法部署与优化也进行了不少工作，不过由于年代久远而且当时没有详尽的记录优化内容，所以这里不去深究细节优化。但是在实际的算法部署过程中，在不同的场景下遇到了不同的问题，没有一个算法是可以适配任何场景的，需要的就是开发人员能够针对不同的场景进行对应的优化，解决场景适配问题。  
&emsp;&emsp;在实际的开发过程中，针对不同的场景，目标跟踪会受到不同的挑战，诸如文章开头提到的外观变形，光照变化，快速运动和运动模糊，背景相似干扰，平面外旋转，平面内旋转，尺度变化，遮挡和出视野等情况。在解决这些问题的过程中，总结技术创新点，并将这些技术创新点形成专利。在整个目标跟踪项目研发过程中，申请了四个发明专利，目前均处于实质审查阶段。在这里不禁又要吐嘈下公司的办事效率，发明专利的申请流程整整在公司内部拖了一年，始终无人问津，如果不是自己去跟踪，估计还要再拖好久。Orz...  
&emsp;&emsp;主要优化的内容有：引入背景信息和自适应回归标签改善跟踪性能，引入显著性检测提高复杂背景下的跟踪性能，提出一种快速的显著性检测方法等等。比较有意思的工作个人认为是利用颜色标签快速实现显著性检测的相关工作，虽然原理简单，但是对复杂背景下的目标跟踪性能提升是明显的。  
&emsp;&emsp;现有的KCF算法需要从目标框中提取图像特征，而这些特征往往是诸如hog特征，颜色特征等，对目标图像的表达能力有限。当在一些背景复杂的使用场景中，往往背景的图像纹理信息表达比目标本身更强烈，影响目标特征的提取，更有甚者，将特征表达强烈的图像背景当作需要跟踪的目标，而真实需要跟踪的目标却由于特征表达较弱被当成背景，导致跟踪失败。在一些背景复杂的场景中，这种现象是普遍存在的。因此，如果能够引入一种破坏背景特征，突出目标特征的图像处理技术，就能够从另一维度告诉机器什么是目标，什么是背景，让机器正确跟踪目标。能让人直观联想到的技术便是显著性检测技术，本质就是抠图技术，从图片中将目标提取出来。但是在实验过程中，显著性技术本身往往实现成本更高，同样配置下实现显著性检测的时间比目标跟踪算法本身更多，不利于产品化。基于此，需要构建一种快速的，低计算成本的目标显著性检测方法来解决该问题。  
&emsp;&emsp;从某些显著性检测原理出发，假设目标图像框最外围像素为背景，基于这些背景由外向内寻找相似像素，通过破坏这些与背景相似的像素便可以达到目的。因此，提出了一种利用颜色标签技术实现快速显著性检测的方法。  
&emsp;&emsp;具体方案如下：  
&emsp;&emsp;颜色标签，顾名思义就是将颜色空间中的所有颜色固定分成几个颜色种类，给每个颜色种类分配一个颜色标签。原图如下：  

<div align=center>
<img src=https://s2.ax1x.com/2019/04/05/ARX7kj.png width=250/>
</div>  

&emsp;&emsp;在Lab颜色空间中将颜色分为$N$个等级，每个颜色等级赋予一个颜色标签数值1~$N$，即分成$N$种颜色，用$N$种颜色代表Lab颜色空间中的所有颜色。  
&emsp;&emsp;对目标框图像进行子区域划分，分成若干大小均等的子区域，每个子区域大小为$m*n$，尽量使每个子区域的大小保持一致。  

<div align=center>
<img src=https://s2.ax1x.com/2019/04/05/ARXbhn.png width=250/>
</div>  

&emsp;&emsp;对图片进行颜色空间转换，由RGB颜色空间转换为Lab颜色空间。分别统计每个子区域图像内的像素均值，并且将均值与所标定的$N$种颜色标签的颜色值进行欧式距离计算，选取欧式距离最小的颜色值所对应的颜色标签作为当前子区域对应的颜色标签，进而形成颜色标签图。  

<div align=center>
<img src=https://s2.ax1x.com/2019/04/05/ARXjXT.png width=250/>
</div>  

&emsp;&emsp;假设颜色标签图最外围的子区域为背景。  

<div align=center>
<img src=https://s2.ax1x.com/2019/04/05/ARXXcV.png width=250/>
</div>  

&emsp;&emsp;从最外围的每个子区域出发，通过查找子区域之间的连通性，从边界往里搜索，逐个子区域进行扩张，直至查找到与外围子区域相连的所有连通域。  

<div align=center>
<img src=https://s2.ax1x.com/2019/04/05/ARXO10.png width=250/>
</div>  

&emsp;&emsp;消除所有查找到的连通域，得到只含目标信息的颜色标签图。  

<div align=center>
<img src=https://s2.ax1x.com/2019/04/05/ARjS74.png width=250/>
</div>  

&emsp;&emsp;将颜色标签图缩放至原始目标矩形框大小。  

<div align=center>
<img src=https://s2.ax1x.com/2019/04/05/ARj9AJ.png width=250/>
</div>  

&emsp;&emsp;将目标信息突出的图像与原图像画面进行与操作，得到去除背景图像后的只保留目标信息的原图像作为结果。  

<div align=center>
<img src=https://s2.ax1x.com/2019/04/05/ARXzBF.png width=250/>
</div>  

&emsp;&emsp;通过上述方法，通过建立颜色标签图的方式，能够快速破坏目标框中的背景信息，突出目标特征。在一些场景的实测过程中，发现这种方法能够有效地解决背景复杂导致目标跟踪失败的问题，而且时间开销极少，不影响目标跟踪算法运行，满足实时性要求。  
&emsp;&emsp;由于篇幅限制，暂不对其他的一些优化方法进行展开，最后贴一个目标跟踪算法实测的魔性gif图。  

<div align=center>
<img src=https://s2.ax1x.com/2019/04/05/ARjAc6.gif width=250/>
</div>  
<div align=center>
摩擦，摩擦，似魔鬼的步伐～
</div>  