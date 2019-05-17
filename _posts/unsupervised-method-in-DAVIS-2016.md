
---

layout:     post
title:      unsupervised method in DAVIS dataset
subtitle:   AGS,MotAdapt,LSMO
date:       2019-05-17
author:     HTF
header-img: img/image.jpg
catalog: true
tags:
    - unsupervised method
    - segmentation
---


# Unsupervised Method in davis2016

## Learning to Segment Moving Objects


���ǽ�����Ϊѧϰ�����ƶ�������������������ǵĿ�ܣ�
- һ��֮֡��Ķ��������˶����䲹�����ʶ��
- ������ۣ���������У���˶������еĴ���
- ʱ��һ���ԣ���Էָ�ʩ���˶����Լ����

![screenShot.png](https://i.loli.net/2019/05/15/5cdc0cc29406226341.png)

�ڱ����У����ǲ����˻��ھ��GRU��ConvGRU�����Ӿ�����ģ�飬����������һ����Ч�ķ�����������Ƶ�ж����ʱ���ݱ��Խ��зָΪ�˳����������Ƶ�����е�����֡������˫��Ӧ�õݹ�ģ��

Motion��������ṹ��

![screenShot.png](https://i.loli.net/2019/05/15/5cdc0cc29406226341.png)

ConvGRU Visual Memory Module�� RNN���䲿��

![1.png](https://i.loli.net/2019/05/15/5cdc156ec034048153.png)

ʵ������ DAVIS �޼ල�������Ƚ�����˼��һ������ǹ������벻ͬ�Ľ�������Ablation study

![screenShot.png](https://i.loli.net/2019/05/15/5cdc18e0d642214675.png)

![screenShot.png](https://i.loli.net/2019/05/15/5cdc1906bb38517610.png)


## Learning Unsupervised Video Object Segmentation through Visual Attention

����**�״�**������֤������۲����Ӿ�ע����Ϊ�ĸ߶�һ���ԣ��������ڶ�̬�����������Ĺ۲����������ע��������ȷ����Ҫ�����ж�֮����ں�ǿ������ԡ����ǽ��޼ල����ƵĿ��ָ��Ϊ����������
- ʱ�����е�UVOS�����Ķ�̬�Ӿ�ע��Ԥ��(DVAP)
- �ռ����е�ע�������Ķ���ָ�(AGOS)
��ô�������˿����� ���ȿ�������Ȥ���� Ȼ��ϸ���۲�

�����㷨��������Ҫ�ŵ㣺
1. ģ�黯��ѵ������ʹ�ð������Ƶ�ָ�ע�ͣ�����ʹ�ø�ʵ�ݵĶ�̬�̶�������ѵ����ʼ��Ƶע��ģ�飬��ʹ�����е�**�̶��ָ���Ծ�̬/ͼ������**   ~~˵�Ĳ�����COCO?~~  ��ѵ�������ָ�ģ��; 
2. ͨ����Դѧϰ����ȫ���ǰ�����; 
3. ��������ѧ�Ϳ�������ע�����Ķ���ɽ����ԡ�

![screenShot.png](https://i.loli.net/2019/05/15/5cdc2394dc2e033234.png)

�����ע����Ԥ���õ���`salient object segmentation�����Էָ�`

- DVAP�õ���CNN-convLSTM�ṹ
- AGOS�õ���FCN�ṹ


### �����Ӿ����ӵ����ݼ�
![screenShot.png](https://i.loli.net/2019/05/16/5cdd173d1f8d266494.png)

### ʵ����
![screenShot.png](https://i.loli.net/2019/05/16/5cdd18e7a684c38849.png)


## Video Object Segmentation using Teacher-Student Adaptation in a Human Robot Interaction (HRI) Setting

MotAdapt�����һ����ӱ��ʦ��ѧϰ���������ڽ��ڻ�������Χ������˫����������ۡ���ʦ�������ṩα��ǩ����Ӧ��ۡ�ѧ�������硣ѧ�������ܹ������������ж���ѧϰ�Ķ�����зֶΣ����������Ǿ�̬���Ƕ�̬��

���ǻ������˾�����Ƶ����ݼ��������ݼ����������HRI���ã���ʾΪ��I��nteractive��V��ideo��O��bject��S��egmentation�����ǵ�IVOS���ݼ�������ͬ����Ͳ�������Ľ�ѧ��Ƶ���������������Ӧ��������DAVIS��FBMS��״̬���ֱ�ΪF-measure��6.8����1.2��������IVOS���ݼ���baseline���������ƣ�mIoUΪ46.1����25.9��

��Ԥ���ڼ䣬���ǵķ�������ѧϰ�ָ����������ֶ��ֶ�ע�͡���ʦģ����һ�������ľ�����磬������˶�����ۣ���ʾΪ���˶�+��ۡ�����Ӧ��ѧ��ģ���ǵ��������ȫ������磬��ʾΪ����ۡ����ڽ�ʦ����������˶������������������Ӧѧ�������α��ǩ

��ʦ���磺RESNET+�ն����
ʹ��α��ǩ��ʦ����Ӧ Teacher-Student Adaptation using Pseudo-labels�����ǵķ����ṩ�����ֲ�ͬ����Ӧ������������ɢ��������ǩ���е���
- ��ʹ����ɢ��ǩʱ�����������Խ�ʦ��������е��������ص�α��ǩ�����ַ����ṩ�˸߾��ȣ������Ե���ȷ����Щ�������صĲ���Ϊ����
- ������ǩ��Ӧ���÷��������˶��κγ�������������Ҫ���������˽��;��ȵĳɱ�

![screenShot.png](https://i.loli.net/2019/05/16/5cdd253ddae5190225.png)

��ɢ���������㷨����
- ������ֱ�Ӱ�teacher�����student��
- ��ɢ��������Ϊ�Ƚ�ȷ�������������ر�Ϊ1��ȷ���ı������ر�Ϊ0�����Կ�������������������ֵ��������Ĳ�����Ҫ�������Ը��鷳

![screenShot.png](https://i.loli.net/2019/05/16/5cdd2c714dee551842.png)



��ƪ���»������һ��[���ݼ�](https://msiam.github.io/ivos/)
