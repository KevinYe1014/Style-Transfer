import functools
from 传智播客.StyleTransfer.src import vgg, transform
import time
import tensorflow as tf, numpy as np, os
from 传智播客.StyleTransfer.src.utils import get_img

STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')  # 要取出来的特征图
CONTENT_LAYER = 'relu4_2'  # 两个loss style和content的
DEVICES = 'CUDA_VISIBLE_DEVICES'


def optimize(content_targets, style_target, conten_weight, style_weight,
             tv_weight, vgg_path, epochs=2, print_iterations=1000,
             batch_size=4, save_path='saver/fns.ckpt', slow=False,
             learning_rate=1e-3, debug=False):
    if slow:
        batch_size = 1


    # 让每一个batch都是四个，如果取出来batch是7个，则将最后三个去掉。
    mod = len(content_targets) % batch_size
    if mod > 0:
        print('Train set has been trimmed slightly ..')
        content_targets = content_targets[:-mod]

    style_features = {}
    batch_shape = (batch_size, 256, 256, 3)  # 训练数据
    style_shape = (1,) + style_target.shape  # style的

    with tf.Graph().as_default(), tf.device('/cpu:0'), tf.Session() as sess:
        style_image = tf.placeholder(tf.float32, shape=style_shape, name='style_image')
        style_image_pre = vgg.preprocess(style_image)  # 预处理
        net = vgg.net(vgg_path, style_image_pre)

        style_pre = np.array([style_target])
        for layer in STYLE_LAYERS:
            features = net[layer].eval(feed_dict={style_image: style_pre})
            features = np.reshape(features, (-1, features.shape[3]))  # 以特征图为单位，原来是[b,h,w,c]
            # print(features.shape)
            # 不是直接以特征图比较的，是以gama 也就是gram比较的。
            gram = np.matmul(features.T, features) / features.size
            style_features[layer] = gram


    with tf.Graph().as_default(), tf.Session() as sess:
        X_content = tf.placeholder(tf.float32, shape=batch_shape, name='X_content')
        X_pre = vgg.preprocess(X_content)

        # precompute content features
        content_features = {}
        content_net = vgg.net(vgg_path, X_pre)
        content_features[CONTENT_LAYER] = content_net[CONTENT_LAYER]

        if slow:
            preds = tf.Variable(tf.random_normal(X_content.get_shape()) * 0.256)
            preds_pred = preds
        else:
            preds = transform.net(X_content / 255.0)
            preds_pred = vgg.preprocess(preds)
        net = vgg.net(vgg_path, preds_pred)

        content_size = _tensor_size(content_features[CONTENT_LAYER]) * batch_size  # batch_size 4

        assert _tensor_size(content_features[CONTENT_LAYER]) == _tensor_size(net[CONTENT_LAYER])
        content_loss = conten_weight * (2 * tf.nn.l2_loss(
            net[CONTENT_LAYER] - content_features[CONTENT_LAYER]) / content_size
                                        )
        # 一张图经过transfer net 和不经过transfer net 然后经过vgg提取特征的
        # 两个特征图的差异是多少

        style_losses = []
        for style_layer in STYLE_LAYERS:
            # 下面net 是两个 经过了transfre net和vgg net
            layer = net[style_layer]
            # batch_size filters相当于channels
            bs, height, width, filters = map(lambda i: i.value, layer.get_shape())  # 自动拆包
            size = height * width * filters  # 图的大小
            feats = tf.reshape(layer, (bs, height * width, filters))
            feats_T = tf.transpose(feats, perm=[0, 2, 1])
            grams = tf.matmul(feats_T, feats) / size # 特征
            # 下面是一个 只经过vgg net的 不用生成网络的gama值是多少
            style_gram = style_features[style_layer]
            style_losses.append(2 * tf.nn.l2_loss(grams - style_gram) / style_gram.size) # 去除 size的影响，所以除以size
        # 很多style_loss值   functools.reduce（定义一个操作，tf.add）
        style_loss = style_weight * functools.reduce(tf.add, style_losses) / batch_size

        # total variation denoising
        # tv_loss 作用是去噪声的
        tv_y_size = _tensor_size(preds[:, 1:, :, :])
        tv_x_size = _tensor_size(preds[:, :, 1:, :])
        y_tv = tf.nn.l2_loss(preds[:, 1:, :, :] - preds[:, :batch_shape[1] - 1, :, :])
        x_tv = tf.nn.l2_loss(preds[:, :, 1:, :] - preds[:, :, :batch_shape[2] - 1, :])
        tv_loss = tv_weight * 2 * (x_tv / tv_x_size + y_tv / tv_y_size) / batch_size

        loss = content_loss + style_loss + tv_loss

        # overall loss
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        sess.run(tf.global_variables_initializer())
        # import random
        # uid = random.randint(1, 100)
        # print('UID: %s' % uid)


        # restore模型
        saver = tf.train.Saver()
        model = os.path.dirname(save_path) + '/'
        checkpoint = tf.train.get_checkpoint_state(model)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print('读入已经训练模型：%s ...' % os.path.abspath(checkpoint.model_checkpoint_path))
        else:
            print('%s 路径下没有已经训练好的模型，开始训练 ...' % os.path.abspath(model))

        for epoch in range(epochs):
            num_examples = len(content_targets) # 样本有多少个
            iterations = 0
            while iterations * batch_size < num_examples:
                start_time = time.time()
                curr = iterations * batch_size # 当前拿出来多少个样本
                step = curr + batch_size
                X_batch = np.zeros(batch_shape, dtype=np.float32)
                for j, img_p in enumerate(content_targets[curr:step]):
                    X_batch[j] = get_img(img_p, (256, 256, 3)).astype(np.float32)

                iterations += 1
                assert X_batch.shape[0] == batch_size
                feed_dict = {X_content: X_batch}

                train_step.run(feed_dict=feed_dict)
                end_time = time.time()
                delta_time = end_time - start_time

                # if debug:
                #     print('UID: %s, batch time: %s' % (uid, delta_time))

                # 下面进行打印
                is_print_iter = int(iterations) % print_iterations == 0
                if slow:
                    is_print_iter = epoch % print_iterations == 0

                is_last = epoch == epochs - 1 and iterations * batch_size >= num_examples
                should_print = is_print_iter or is_last

                print("Epoch: {}, iteration: {}".format(epoch, iterations))
                if should_print:
                    to_get = [style_loss, content_loss, tv_loss, loss, preds]  # preds 最终输出的图像结果
                    test_dict = {X_content: X_batch}
                    tup = sess.run(to_get, feed_dict=test_dict)
                    _style_loss, _content_loss, _tv_loss, _loss, _preds = tup
                    losses = (_style_loss, _content_loss, _tv_loss, _loss)
                    if slow:
                        _preds = vgg.unprocess(_preds)
                    else:
                        # pass
                        res = saver.save(sess, save_path, global_step=iterations)

                    yield (_preds, losses, iterations, epoch)
                    # yield返回和return返回 可以通俗一点理解是，yield返回不会终止，但是return返回会终止。


def _tensor_size(tensor):
    from operator import mul
    return functools.reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)
