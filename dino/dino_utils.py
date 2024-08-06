import math
import os
import datetime
import re
import warnings
import numpy as np

import tensorflow as tf
import tensorflow_addons as tfa
from vit_keras import layers
import typing_extensions as tx

import utils as utils

ConfigDict = tx.TypedDict(
    "ConfigDict",
    {
        "dropout": float,
        "mlp_dim": int,
        "num_heads": int,
        "num_layers": int,
        "hidden_size": int,
    },
)

class MultiCropWrapper(tf.keras.models.Model):
    def __init__(self, backbone, head, weights=None, hiddenSize=768, patchSize = None, flattenBackbone = False, reducedGanFilters = None):
        super(MultiCropWrapper, self).__init__()
        self.head = head
        self.backbone = backbone
        self.hiddenSize = hiddenSize
        self.patchSize = patchSize
        self.attentionPredictor = None
        self.flattenBackbone = flattenBackbone
        self.reducedGanFilters = reducedGanFilters

        if self.flattenBackbone:
            self.flattenLayer = tf.keras.layers.Flatten()

        if self.reducedGanFilters is not None:
            self.ReduceGanFeaturesLayer = tf.keras.layers.Conv2D(filters=reducedGanFilters,kernel_size=1,strides=1,padding="valid")

        if weights:
            try:
                print("Restoring model weights from: ", weights)
                self.load_weights(weights)
            except Exception:
                raise ValueError

    def call(self, x):
        output = tf.zeros((0, self.hiddenSize), dtype=x.dtype)
        _out = self.backbone(x)

        if self.reducedGanFilters is not None:
            _out = self.ReduceGanFeaturesLayer(_out[0])
            _out = self.flattenLayer(_out)
        elif self.flattenBackbone:
            _out = self.flattenLayer(_out[0])
            
        if isinstance(_out, tuple):
            _out = _out[0]
        output = tf.concat([output, _out], axis=0)
        return self.head(output)
    
    def predictEmbeddingBatch(self,batch):
        batch = tf.convert_to_tensor(batch)

        embedding = self.backbone(batch)
        if self.reducedGanFilters is not None:
            embedding = self.ReduceGanFeaturesLayer(embedding[0])
            embedding = self.flattenLayer(embedding)
        elif self.flattenBackbone:
            embedding = self.flattenLayer(embedding[0])
        predictions = self.head(embedding)

        return predictions

    def predictEmbedding(self, dataX):
        if isinstance(dataX,tf.keras.utils.Sequence):
            predictions = []
            for batch in dataX:
                batchPredictions = self.predictEmbeddingBatch(batch)
                predictions.extend(batchPredictions)
        else:
            predictions = self.predictEmbeddingBatch(dataX)
        return predictions

    def predictAttentionBatch(self,batchX):
        batchX = tf.convert_to_tensor(batchX)
        batchAttention = self.backbone.predictAttentionMap(batchX)
        
        numberOfHeads = batchAttention.shape[1]
        flatBatchAttentions = tf.reshape(batchAttention[:,:,0,1:], (batchAttention.shape[0],numberOfHeads,-1))
        return batchX, flatBatchAttentions

    def predictAttentionMap(self,data, nImages=None):
        if isinstance(data,tf.keras.utils.Sequence):
            flatAttentionMaps = []
            dataX = []
            if nImages is None: nImages = len(data)*data.batch_size
            for batch in data:
                if len(flatAttentionMaps) >= nImages:
                    break
                batchX, flatBatchAttentions = self.predictAttentionBatch(batch)
                dataX.extend(batchX)
                flatAttentionMaps.extend(flatBatchAttentions)
        else:
            dataX, flatAttentionMaps = self.predictAttentionBatch(data)
        return flatAttentionMaps, dataX

class WeightNorm(tf.keras.layers.Wrapper):
    """Layer wrapper to decouple magnitude and direction of the layer's weights.

    This wrapper reparameterizes a layer by decoupling the weight's
    magnitude and direction. This speeds up convergence by improving the
    conditioning of the optimization problem. It has an optional data-dependent
    initialization scheme, in which initial values of weights are set as functions
    of the first minibatch of data. Both the weight normalization and data-
    dependent initialization are described in [Salimans and Kingma (2016)][1].

    #### Example

    ```python
    net = WeightNorm(tf.keras.layers.Conv2D(2, 2, activation='relu'),
            input_shape=(32, 32, 3), data_init=True)(x)
    net = WeightNorm(tf.keras.layers.Conv2DTranspose(16, 5, activation='relu'),
                        data_init=True)
    net = WeightNorm(tf.keras.layers.Dense(120, activation='relu'),
                        data_init=True)(net)
    net = WeightNorm(tf.keras.layers.Dense(num_classes),
                        data_init=True)(net)
    ```

    #### References

    [1]: Tim Salimans and Diederik P. Kingma. Weight Normalization: A Simple
        Reparameterization to Accelerate Training of Deep Neural Networks. In
        _30th Conference on Neural Information Processing Systems_, 2016.
        https://arxiv.org/abs/1602.07868
    """
    def __init__(self, layer, data_init=True, trainableG = True, **kwargs):
        """Initialize WeightNorm wrapper.

        Args:
            layer: A `tf.keras.layers.Layer` instance. Supported layer types are
            `Dense`, `Conv2D`, and `Conv2DTranspose`. Layers with multiple inputs
            are not supported.
            data_init: `bool`, if `True` use data dependent variable initialization.
            **kwargs: Additional keyword args passed to `tf.keras.layers.Wrapper`.

        Raises:
            ValueError: If `layer` is not a `tf.keras.layers.Layer` instance.

        """
        if not isinstance(layer, tf.keras.layers.Layer):
            raise ValueError(
                'Please initialize `WeightNorm` layer with a `tf.keras.layers.Layer` '
                'instance. You passed: {input}'.format(input=layer))

        layer_type = type(layer).__name__
        if layer_type not in ['Dense', 'Conv2D', 'Conv2DTranspose']:
            warnings.warn('`WeightNorm` is tested only for `Dense`, `Conv2D`, and `Conv2DTranspose` layers. You passed a layer of type `{}`'.format(layer_type))

        super(WeightNorm, self).__init__(layer, **kwargs)
        self.trainableG = trainableG
        self.data_init = data_init
        self._track_trackable(layer, name='layer')
        self.filter_axis = -2 if layer_type == 'Conv2DTranspose' else -1

    def _compute_weights(self):
        """Generate weights with normalization."""
        # Determine the axis along which to expand `g` so that `g` broadcasts to
        # the shape of `v`.
        new_axis = -self.filter_axis - 3

        # `self.kernel_norm_axes` is determined by `self.filter_axis` and the rank
        # of the layer kernel, and is thus statically known.
        self.layer.kernel = tf.nn.l2_normalize(self.v, axis=self.kernel_norm_axes) * tf.expand_dims(self.g, new_axis)

    def _init_norm(self):
        """Set the norm of the weight vector."""
        kernel_norm = tf.sqrt(tf.reduce_sum(tf.square(self.v), axis=self.kernel_norm_axes))
        self.g.assign(kernel_norm)

    def _data_dep_init(self, inputs):
        """Data dependent initialization."""
        # Normalize kernel first so that calling the layer calculates
        # `tf.dot(v, x)/tf.norm(v)` as in (5) in ([Salimans and Kingma, 2016][1]).
        self._compute_weights()

        activation = self.layer.activation
        self.layer.activation = None

        use_bias = self.layer.bias is not None
        if use_bias:
            bias = self.layer.bias
            self.layer.bias = tf.zeros_like(bias)

        # Since the bias is initialized as zero, setting the activation to zero and
        # calling the initialized layer (with normalized kernel) yields the correct
        # computation ((5) in Salimans and Kingma (2016))
        x_init = self.layer(inputs)
        norm_axes_out = list(range(x_init.shape.rank - 1))
        m_init, v_init = tf.nn.moments(x_init, norm_axes_out)
        scale_init = 1. / tf.sqrt(v_init + 1e-10)

        self.g.assign(self.g * scale_init)
        if use_bias:
            self.layer.bias = bias
            self.layer.bias.assign(-m_init * scale_init)
        self.layer.activation = activation

    def build(self, input_shape=None):
        """Build `Layer`.

        Args:
            input_shape: The shape of the input to `self.layer`.

        Raises:
            ValueError: If `Layer` does not contain a `kernel` of weights
        """

        input_shape = tf.TensorShape(input_shape).as_list()
        input_shape[0] = None
        self.input_spec = tf.keras.layers.InputSpec(shape=input_shape)

        if not self.layer.built:
            self.layer.build(input_shape)

            if not hasattr(self.layer, 'kernel'):
                raise ValueError('`WeightNorm` must wrap a layer that contains a `kernel` for weights')

            kernel_norm_axes = list(range(self.layer.kernel.shape.rank))
            kernel_norm_axes.pop(self.filter_axis)
            # Convert `kernel_norm_axes` from a list to a constant Tensor to allow
            # TF checkpoint saving.
            self.kernel_norm_axes = tf.constant(kernel_norm_axes)

            self.v = self.layer.kernel

            # to avoid a duplicate `kernel` variable after `build` is called
            self.layer.kernel = None
            self.g = self.add_weight(name='g',shape=(int(self.v.shape[self.filter_axis]),),initializer='ones',dtype=self.v.dtype,trainable=self.trainableG)

            if self.trainableG:
                self.initialized = self.add_weight(name='initialized',dtype=tf.bool,trainable=False)
                self.initialized.assign(False)

        super(WeightNorm, self).build()

    @tf.function
    def call(self, inputs):
        """Call `Layer`."""
        if self.trainableG:
            if not self.initialized:
                if self.data_init:
                    self._data_dep_init(inputs)
                else:
                    # initialize `g` as the norm of the initialized kernel
                    self._init_norm()

                self.initialized.assign(True)

        self._compute_weights()
        output = self.layer(inputs)
        return output

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(self.layer.compute_output_shape(input_shape).as_list())

class DinoHead(tf.keras.models.Model):
    def __init__(self,in_dim, out_dim, use_bn=False,norm_last_layer=True,nlayers=3,hidden_dim=2048,bottleneck_dim=256,):
        super(DinoHead, self).__init__()
        self.in_dim = in_dim
        self.use_bn = use_bn
        self.out_dim = out_dim
        self.nlayers = nlayers
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.norm_last_layer = norm_last_layer

        # Contstruct mlp block
        layer = []
        layer.append(tf.keras.layers.Dense(self.hidden_dim, input_shape=(self.in_dim,)))
        if self.use_bn:
            layer.append(tf.keras.layers.BatchNormalization())
        layer.append(tfa.layers.GELU())
        
        for _ in range(self.nlayers - 2):
            layer.append(tf.keras.layers.Dense(self.hidden_dim))
            if self.use_bn:
                layer.append(tf.keras.layers.BatchNormalization())
            layer.append(tfa.layers.GELU())
        
        layer.append(tf.keras.layers.Dense(self.bottleneck_dim))
        self.mlp_block = tf.keras.Sequential(layer)

        trainableG = norm_last_layer == False
        self.last_layer = WeightNorm(tf.keras.layers.Dense(self.out_dim, use_bias=False), data_init=False, trainableG=trainableG)

    def call(self, input_tensor, training=None):
        x = self.mlp_block(input_tensor, training)
        x = tf.nn.l2_normalize(x, axis=-1)
        x = self.last_layer(x)
        return x
    
class Dino(tf.keras.models.Model):
    def __init__(self, dinoSettings, dataSettings, teacher_model, student_model, batchSize, customPositionalEncoding=None,metadataEncoding=None, momentumScheduler = None,nBatches=None,teacherTempScheduler=None):
        super(Dino, self).__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model

        self.customPositionalEncoding = customPositionalEncoding
        self.metadataEncoding = metadataEncoding
        self.nGlobalCrops = 2
        self.nLocalCrops = dinoSettings['nLocalCrops']
        self.momentumScheduler = momentumScheduler
        self.nBatches = nBatches
        self.teacherTempScheduler = teacherTempScheduler

        self.ncrops = self.nGlobalCrops+self.nLocalCrops
        self.student_temp = 0.1
        self.center_momentum = 0.9
        self.freezeLastLayerEpochs = dinoSettings['freeze_last_layer']
        self.normLastLayer = dinoSettings['normLastLayer']
        self.clipGradient = dinoSettings['clipGradient']

        self.center = tf.Variable(tf.zeros((self.nGlobalCrops*batchSize,dinoSettings['outputDim'])), trainable=False, dtype=tf.float32)
        self.acculatedInitialized = tf.Variable(False, trainable=False, dtype=tf.bool)

    def compile(self, optimizer):
        super(Dino, self).compile()
        self.optimizer = optimizer

    def train_step(self, data):
        iteration = self.optimizer.iterations
        epoch = iteration // self.nBatches
        teacherTemp = self.teacherTempScheduler[epoch]

        global_image, local_image = data
        global_image = sum(global_image, ())
        local_image = sum(local_image, ())
        global_image = tf.stack(global_image)
        local_image = tf.stack(local_image)

        with tf.GradientTape() as tape:
            teacher_output = self.teacher_model(global_image)
            student_global_output = self.student_model(global_image)
            student_local_output = self.student_model(local_image)
            student_output = tf.concat([student_global_output, student_local_output], axis=0)
        
            student_output = student_output / self.student_temp
            student_output = tf.split(student_output, num_or_size_splits=self.ncrops)

            teacher_out = tf.stop_gradient(tf.nn.softmax((teacher_output - self.center) / teacherTemp, axis=-1)) # Shape = 16x256    OK
            teacher_out = tf.split(teacher_out, num_or_size_splits=2) # [8x256]*2           OK

            total_loss = 0
            n_loss_terms = 0
            for idx, q in enumerate(teacher_out):
                for v in range(len(student_output)): # global 0 with local 1 and global 1 with local 0 are still being processed
                    q = tf.stop_gradient(q)
                    if v == idx:
                        continue
                    loss = tf.reduce_sum(-q * tf.nn.log_softmax(student_output[v], axis=-1), axis=-1)
                    total_loss += tf.math.reduce_mean(loss)
                    n_loss_terms += 1
            total_loss /= n_loss_terms #/18
                
            loss = tf.reduce_mean(total_loss)
            
            student_gradients = tape.gradient(loss, self.student_model.trainable_variables)
            studentGradientsNorm = tf.linalg.global_norm(student_gradients)
            gradientMin = tf.reduce_min([tf.reduce_min(grad) for grad in student_gradients])
            gradientMax = tf.reduce_max([tf.reduce_max(grad) for grad in student_gradients])

        if self.clipGradient != 0:
            student_gradients, _ = tf.clip_by_global_norm(student_gradients, self.clipGradient)
        self.optimizer.apply_gradients(zip(student_gradients, self.student_model.trainable_variables))
   
        # 4x256
        batch_center = tf.math.reduce_sum(teacher_output, axis=0)
        batch_center = batch_center / teacher_output.shape[0]
        self.center.assign(tf.multiply(self.center, self.center_momentum) + tf.multiply(batch_center, (1 - self.center_momentum))) 

        # EMA update for the teacher
        momentum=self.momentumScheduler(iteration)

        # Assign student to teacher
        if self.normLastLayer:
            for param_q, param_k in zip(self.student_model.weights, self.teacher_model.weights):
                param_k.assign(param_k * momentum + (1 - momentum) * param_q)
        else:
            for param_q, param_k in zip(self.student_model.weights, self.teacher_model.weights):
                if param_q.dtype == tf.bool:
                    continue
                param_k.assign(param_k * momentum + (1 - momentum) * param_q)
            
        learningRate = self.optimizer.lr(iteration)
        studentTotal = tf.reduce_sum([tf.reduce_sum(var) for var in self.student_model.backbone.weights])
        teachterTotal = tf.reduce_sum([tf.reduce_sum(var) for var in self.teacher_model.backbone.weights])
        return {"loss": loss, 'm':momentum, 'sum student': studentTotal, 'sum teacher ': teachterTotal, 'iteration': iteration, 'temp':teacherTemp, 'lr':learningRate, 'gradNorm':studentGradientsNorm, 'gradMin':gradientMin, 'gradMax':gradientMax}

    def test_step(self, data):
        iteration = self.optimizer.iterations
        epoch = iteration // self.nBatches
        teacherTemp = self.teacherTempScheduler[epoch]

        if self.customPositionalEncoding is not None or self.metadataEncoding is not None:
            (dataX, metadata) = data
            global_image, local_image = dataX
            globalMetadata, localMetadata = metadata
        else:
            global_image, local_image = data

        local_image = sum(local_image, ())
        global_image = sum(global_image, ())
        local_image = tf.stack(local_image)
        global_image = tf.stack(global_image)


        if self.customPositionalEncoding is not None or self.metadataEncoding is not None:
            teacher_output = self.teacher_model((global_image, globalMetadata),training=False)
            student_output = self.student_model((local_image, localMetadata),training=False)
        else:
            teacher_output = self.teacher_model(global_image,training=False)
            student_global_output = self.student_model(global_image,training=False)
            student_local_output = self.student_model(local_image,training=False)
            student_output = tf.concat([student_global_output, student_local_output], axis=0)

        #teacher_output = tf.cast(teacher_output, tf.float32) # 16x256
        #student_output = tf.cast(student_output, tf.float32) # 80x256 
    
        # Batch size = 4. 8 local and 2 global crops
        student_out = student_output / self.student_temp
        student_out = tf.split(student_out, num_or_size_splits=self.ncrops) # [8x256]*10

        teacher_out = tf.stop_gradient(tf.nn.softmax((teacher_output - self.center) / teacherTemp, axis=-1)) # Shape = 16x256
        teacher_out = tf.split(teacher_out, num_or_size_splits=2) # [8x256]*2           OK

        total_loss = 0
        n_loss_terms = 0
        for idx, q in enumerate(teacher_out):
            for v in range(len(student_out)): # global 0 with local 1 and global 1 with local 0 are still being processed
                q = tf.stop_gradient(q)
                if v == idx:
                    continue
                loss = tf.reduce_sum(-q * tf.nn.log_softmax(student_out[v], axis=-1), axis=-1)
                total_loss += tf.math.reduce_mean(loss)
                n_loss_terms += 1
        total_loss /= n_loss_terms #/18
            
        loss = tf.reduce_mean(total_loss)
        
        return {"loss": loss}

    def call(self, data):
        if self.customPositionalEncoding is not None or self.metadataEncoding is not None:
            (dataX, metadata) = data
            global_image, local_image = dataX
            globalMetadata, localMetadata = metadata
        else:
            global_image, local_image = data
        
        global_image = tf.convert_to_tensor(global_image)
        local_image = tf.convert_to_tensor(local_image)

        local_image = tf.reshape(local_image, (local_image.shape[0]*local_image.shape[1],)+local_image.shape[2:])
        global_image = tf.reshape(global_image, (global_image.shape[0]*global_image.shape[1],) + global_image.shape[2:])

        if self.customPositionalEncoding is not None or self.metadataEncoding is not None:
            teacher_output = self.teacher_model([global_image,globalMetadata], training=False)
        else:
            teacher_output = self.teacher_model(global_image, training=False)
        return teacher_output

class VitModel(tf.keras.models.Model):
    """Adds (optionally learned) positional embeddings to the inputs."""
    def __init__(self, globalImageSize, localImageSize, patch_size: int,num_layers: int,hidden_size: int,num_heads: int,name: str,mlp_dim: int,dropout=0.1,nChannels = 3, changeFlatteningOrder = False, positionEmbeddingFunction = None):
        super(VitModel, self).__init__()

        self.changeFlatteningOrder = changeFlatteningOrder
        self.hiddenSize = hidden_size
        self.positionEmbeddingFunction = positionEmbeddingFunction
        self.patchSize = patch_size
        self.localImageShape = localImageSize
        nPatches = (globalImageSize[0] // patch_size) * (globalImageSize[1] // patch_size)

        # Split the image in paches with the hidden size as the token length
        self.conv2d = tf.keras.layers.Conv2D(filters=hidden_size,kernel_size=patch_size,strides=patch_size,padding="valid",name="embedding",)
        
        # Create variable for the position embedding based on the global image dimension
        posEmbedInit = tf.random.truncated_normal((1,nPatches+1, self.hiddenSize), stddev=.02)
        self.posEmbed = tf.Variable(posEmbedInit, trainable=True, name="pos_embedding")
        self.classToken = layers.ClassToken(name="class_token")

        # Transformer blocks
        self.blocks = []
        for n in range(num_layers):
            block = layers.TransformerBlock(num_heads=num_heads,mlp_dim=mlp_dim,dropout=dropout,name=f"Transformer/encoderblock_{n}",)
            self.blocks.append(block)

        # Finish the backbone
        self.normLayer = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="Transformer/encoder_norm")
        self.lambdaLayer = tf.keras.layers.Lambda(lambda v: v[:, 0], name="ExtractToken")

    def interpolate_pos_encoding(self, x, image_size):
        # x.shape = 8,197,384
        N = self.posEmbed.shape[1] - 1
        if x.shape[1] == N+1:
            return self.posEmbed
        class_pos_embed = self.posEmbed[:, 0]
        patch_pos_embed = self.posEmbed[:, 1:]
        dim = x.shape[-1]

        w0 = image_size[2] // self.patchSize
        h0 = image_size[1] // self.patchSize

        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        #w0, h0 = w0 + 0.1, h0 + 0.1
        reshapedPatchPosEmb = tf.keras.layers.Reshape((int(math.sqrt(N)), int(math.sqrt(N)), dim))(patch_pos_embed)
        # 1,14,14,384
        resizedEmbedding = tf.image.resize(reshapedPatchPosEmb,size=(w0,h0),method='bicubic')
        # 1,6,6,384
        #assert int(w0) == resizedEmbedding.shape[1] and int(h0) == resizedEmbedding.shape[2]
        patch_pos_embed = tf.keras.layers.Reshape((-1, dim))(resizedEmbedding)
        # 1,36,384
        classPosEmbedding = tf.expand_dims(class_pos_embed,axis=0)#tf.cast(tf.expand_dims(class_pos_embed,axis=0),dtype=x.dtype)
        return tf.concat((classPosEmbedding, patch_pos_embed), axis=1)

    def call(self, inputs):
        x = self.conv2d(inputs)

        if self.changeFlatteningOrder:
            x = tf.keras.layers.Permute((2, 1, 3))(x)

        x = tf.keras.layers.Reshape((-1, self.hiddenSize))(x)
        x = self.classToken(x)

        image_size = tf.shape(inputs)
        posEncoding = self.interpolate_pos_encoding(x,image_size)
        assert x.shape [1:]== posEncoding.shape[1:]
        x = x + posEncoding# tf.cast(posEncoding,dtype = x.dtype)

        for block in self.blocks:
            x, _ = block(x)
        
        x = self.normLayer(x)
        x = self.lambdaLayer(x)
        return x
    
    def predictAttentionMap(self, inputs):
        x = self.conv2d(inputs)

        if self.changeFlatteningOrder:
            x = tf.keras.layers.Permute((2, 1, 3))(x)

        x = tf.keras.layers.Reshape((-1, self.hiddenSize))(x)
        x = self.classToken(x)

        image_size = tf.shape(inputs)
        posEncoding = self.interpolate_pos_encoding(x,image_size)
        assert x.shape [1:]== posEncoding.shape[1:]
        x = x + posEncoding# tf.cast(posEncoding,dtype = x.dtype)

        for block in self.blocks[:-1]:
            x, _ = block(x)
        x, weights = self.blocks[-1](x)

        return weights

class CustomCosineScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
        super(CustomCosineScheduler, self).__init__()

        warmup_schedule = np.array([])
        warmup_iters = warmup_epochs * niter_per_ep
        if warmup_epochs > 0:
            warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(epochs * niter_per_ep - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

        self.schedule = tf.cast(tf.concat((warmup_schedule, schedule),axis=0), tf.float32)
        assert len(self.schedule) == epochs * niter_per_ep

    def __call__(self, step):
        value = tf.gather(self.schedule, tf.cast(step, tf.int32))
        return value

class DinoLoader:
    def __init__(self, dinoSettings, dataSettings, ganSettings= None, overideBatchSize = None, resultsDir = None):
        self.dinoSettings = dinoSettings
        self.dataSettings = dataSettings
        self.ganSettings = ganSettings

        # Data settings
        self.inputShape = dataSettings['inputShape']
        if overideBatchSize is None:
            self.batchSize = dataSettings['batchSize']
        else:
            self.batchSize = overideBatchSize
        self.normalizationMethod = dataSettings['normalizationMethod']

        modelSettings = dinoSettings

        self.modelBaseName = modelSettings['modelBaseName']
        self.architecture = modelSettings['architecture']
        self.patchSize = modelSettings['patchSize']
        self.outputDim = modelSettings['outputDim']
        self.teacherGlobalSize = modelSettings['teacherGlobalSize']
        self.studentLocalSize = modelSettings['studentLocalSize']
        self.globalScale = modelSettings['globalScale']
        self.localScale = modelSettings['localScale']
        self.nChannels = modelSettings['nChannels']
        self.nEpochs = modelSettings['nEpochs']
        self.nLocalCrops = modelSettings['nLocalCrops']
        self.hiddenSize = modelSettings['hiddenSize']

        # Construct dino model name
        self.dinoModelName = utils.models.getDinoModelName(modelSettings, dataSettings)

        # Data locations
        if resultsDir is None:
            self.modelDir, self.logLocation = utils.functions.getModelLocation(os.path.join('dino',self.dinoModelName))
            self.tensorboardLocation = self.logLocation
            self.plotLocation = utils.functions.getPlotLocation(self.dataSettings['datasetName'], self.dinoModelName)
        else:
            self.modelDir, self.tensorboardLocation = utils.functions.getModelLocation(os.path.join('dino',self.dinoModelName))
            self.plotLocation = utils.functions.getPlotLocation(resultsDir, self.dinoModelName)
            self.logLocation = self.plotLocation
 
        self.checkpointDir = self.modelDir
        self.logName = tf.Variable("dino_run_{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

    def buildGanEncoderModel(self, image_size, ganConfig,modelType):
        scaleInputShape = None
        if image_size[0] != self.inputShape[0] or image_size[1] != self.inputShape[1]:
            scaleInputShape = (image_size[0],image_size[1],self.inputShape[2])

        ganEncoder = utils.models.Models.Gan_Encoder_v1(self.inputShape,depth=ganConfig['generatorDepth'],scaleInputShape = scaleInputShape)

        dummyIn = np.zeros((self.dataSettings['batchSize'], image_size[0], image_size[1], self.inputShape[2]))
        dummyOut = ganEncoder(dummyIn)
        nFeatures = np.reshape(dummyOut[0], (self.dataSettings['batchSize'],-1)).shape[1]
        return ganEncoder, nFeatures

    def load_base(self, modelType):      
        if modelType == 'teacher':
            image_size = self.dinoSettings['teacherGlobalSize']
        else:
            image_size = self.dinoSettings['studentLocalSize']
        name = self.dinoSettings['architecture']
        patchSize = self.dinoSettings['patchSize']
        nChannels = self.dinoSettings['nChannels']
        changeFlatteningOrder = self.dinoSettings['patchFlatteningSwap']
        customPositionalEncoding = self.dinoSettings['customPositionalEncoding']

        CONFIG_XXS: ConfigDict = { # config copied from DINO 
            "dropout":0.0,      # 0.1,     # Dropout rate transformer blocks
            "mlp_dim":768,     # 1536,    # mlp_dim transformer blocks # when ratio, it is mlp_dim = ratio*hidden_size  ratio=4
            "num_heads": 3,     # 6,    # Number of heads in transformer blocks
            "num_layers": 12,    # 12,   # number of transformer blocks
            "hidden_size": 192      # 384  # Number of filters of conv block and reshape before transformer
        } 

        CONFIG_XS: ConfigDict = {
            "dropout":0.0,          # Dropout rate transformer blocks
            "mlp_dim":1152,         # mlp_dim transformer blocks # when ratio, it is mlp_dim = ratio*hidden_size    ratio=4*288
            "num_heads": 4,         # Number of heads in transformer blocks. Must be dividable hidden size
            "num_layers": 12,       # number of transformer blocks
            "hidden_size": 288      # Number of filters of conv block and reshape before transformer
        }

        CONFIG_S: ConfigDict = {
            "dropout":0.0,          # Dropout rate transformer blocks
            "mlp_dim":1536,         # mlp_dim transformer blocks # when ratio, it is mlp_dim = ratio*hidden_size ratio=4
            "num_heads": 6,         # Number of heads in transformer blocks
            "num_layers": 12,       # number of transformer blocks
            "hidden_size": 384      # Number of filters of conv block and reshape before transformer
        } 
        CONFIG_B: ConfigDict = {
            "dropout": 0.0,
            "mlp_dim": 3072,
            "num_heads": 12,
            "num_layers": 12,
            "hidden_size": 768,
        } 

        CONFIG_L: ConfigDict = {
                "dropout": 0.0,
                "mlp_dim": 4096,
                "num_heads": 16,
                "num_layers": 24,
                "hidden_size": 1024,
        }

        vitConfig = None
        ganConfig=None

        if name == 'vit-xxs':
            vitConfig = CONFIG_XXS
        elif name == 'vit-xs':
            vitConfig = CONFIG_XS
        elif name == 'vit-s':
            vitConfig = CONFIG_S
        elif name == 'vit-b':
            vitConfig = CONFIG_B
        elif name == 'vit-l':
            vitConfig = CONFIG_L
        elif name == 'gan':
            ganConfig = self.ganSettings

        if vitConfig is not None:
            model = VitModel(
                **vitConfig,
                globalImageSize=self.dinoSettings['teacherGlobalSize'],
                name=name,
                patch_size=patchSize,
                localImageSize=self.dinoSettings['studentLocalSize'],
                nChannels = nChannels,
                changeFlatteningOrder = changeFlatteningOrder,
                positionEmbeddingFunction= customPositionalEncoding
            )
            nFeatures = vitConfig['hidden_size']
        elif ganConfig is not None:
            model, nFeatures = self.buildGanEncoderModel(image_size, ganConfig,modelType)

        return nFeatures, model
    
    def buildModel(self, nBatches):
        # Create model
        hiddenSize, teacher = self.load_base('teacher')
        hiddenSize, student = self.load_base('student')
        
        flattenBackbone = self.dinoSettings['architecture'] == 'gan'
        reducedGanDimension = self.dinoSettings['reducedGanDimension']

        if flattenBackbone and reducedGanDimension is None:
            hidden_dim = 512
        else:
            hidden_dim = 2048
        
        if reducedGanDimension is not None:
            encoderOutputSize = (4,16,1024)
            encoderOutputDimension = encoderOutputSize[0] * encoderOutputSize[1]*encoderOutputSize[2]
            if (encoderOutputDimension) != hiddenSize:
                raise Exception("Hidden size does not match hard coded gan encoder output size")
            reducedGanFilters = round((reducedGanDimension/encoderOutputDimension)*encoderOutputSize[-1])
            hiddenSize = encoderOutputSize[0] * encoderOutputSize[1]*reducedGanFilters
        else:
            reducedGanFilters = None

        # teacher has always norm last layer
        studentHead = DinoHead(hiddenSize,self.outputDim,use_bn = self.dinoSettings['use_bn_in_head'], norm_last_layer=self.dinoSettings['normLastLayer'], hidden_dim=hidden_dim)
        teacherHead = DinoHead(hiddenSize,self.outputDim,use_bn = self.dinoSettings['use_bn_in_head'], norm_last_layer=self.dinoSettings['normLastLayer'], hidden_dim=hidden_dim)

        student = MultiCropWrapper(backbone=student, head=studentHead, hiddenSize=hiddenSize, patchSize=self.patchSize,flattenBackbone=flattenBackbone, reducedGanFilters=reducedGanFilters)
        teacher = MultiCropWrapper(backbone=teacher, head=teacherHead, hiddenSize=hiddenSize, patchSize=self.patchSize,flattenBackbone=flattenBackbone, reducedGanFilters=reducedGanFilters)
        
        self.lrScheduler = CustomCosineScheduler(
            self.dinoSettings['learningRate'],
            self.dinoSettings['min_lr'],
            self.dinoSettings['nEpochs'],
            nBatches,
            self.dinoSettings['warmup_epochs']
        )

        if self.dinoSettings['weightDecay'] == self.dinoSettings['weight_decay_end']:
            self.wdScheduler = self.dinoSettings['weightDecay']
        else:
            self.wdScheduler = CustomCosineScheduler(
                self.dinoSettings['weightDecay'],
                self.dinoSettings['weight_decay_end'],
                self.dinoSettings['nEpochs'],
                nBatches
            )
        
        momentumScheduler = CustomCosineScheduler(
            self.dinoSettings['momentumTeacher'], 
            1,
            self.dinoSettings['nEpochs'],
            nBatches
        )

        if self.dinoSettings['warmup_teacher_temp_epochs'] == 0:
            teacherTempScheduler = tf.ones((self.dinoSettings['nEpochs']))*self.dinoSettings['teacher_temp']
        else:
            teacherTempScheduler = tf.concat((tf.linspace(self.dinoSettings['warmup_teacher_temp'], self.dinoSettings['teacher_temp'], self.dinoSettings['warmup_teacher_temp_epochs']),
                                              tf.ones((self.dinoSettings['nEpochs'] - self.dinoSettings['warmup_teacher_temp_epochs'])) * self.dinoSettings['teacher_temp'],),axis=0)   

        if self.dinoSettings['optimizer'] == 'adam':
            # weight decay should be used by the adamW optimizer
            self.optimizer = tf.keras.optimizers.Adam(self.lrScheduler)
        elif self.dinoSettings['optimizer'] == 'adamw':
            if os.environ.get('OS','') == "Windows_NT":
                self.optimizer = tfa.optimizers.AdamW(learning_rate=self.lrScheduler, weight_decay=self.wdScheduler)
            else:
                self.optimizer = tf.keras.optimizers.experimental.AdamW(learning_rate=self.lrScheduler, weight_decay=self.wdScheduler)
        elif self.dinoSettings['optimizer'] == 'lamb':
            self.optimizer = tfa.optimizers.LAMB(learning_rate=self.lrScheduler, weight_decay_rate=self.wdScheduler)
        else:
            raise ValueError("Optimizer not recognized")

        self.model = Dino(self.dinoSettings, self.dataSettings, teacher, student, self.batchSize, self.dinoSettings['customPositionalEncoding'], self.dinoSettings['metadataEncoding'], momentumScheduler, nBatches,teacherTempScheduler)
        self.model.compile(optimizer=self.optimizer)

        fakeGenerator = utils.datasets.Generators.FakeDinoGenerator(self.batchSize, self.nChannels,self.teacherGlobalSize,self.studentLocalSize,2, self.nLocalCrops, 1)
        modelIn = fakeGenerator[0]
                    
        self.model(modelIn)

        for param_q, param_k in zip(self.model.student_model.weights, self.model.teacher_model.weights):
            if param_q.dtype == tf.bool:
                continue
            param_k.assign(param_q)

    def updateCheckpoint(self):
        checkpoint = tf.train.Checkpoint(logName = self.logName, teacherCenter=self.model.center, optimizer = self.model.optimizer, studentModel = self.model.student_model, teacherModel = self.model.teacher_model)
        self.checkpointManager = tf.train.CheckpointManager(checkpoint, self.checkpointDir, max_to_keep=1)

    def loadCheckpoint(self, restoreTraining=True, restoreTeacher = False, loadEpoch = None):
        if restoreTeacher:
            # load Dino weights
            checkpoint = tf.train.Checkpoint(logName = self.logName,teacherModel = self.dinoTeacher)
            checkpointManager = tf.train.CheckpointManager(checkpoint, self.checkpointDir, max_to_keep=1)
            dinoEpoch = 0
            if checkpointManager.latest_checkpoint:
                if loadEpoch is None:
                    checkpointFilename = checkpointManager.latest_checkpoint
                else:
                    checkpointFilename = os.path.join(self.checkpointDir, 'ckpt-{}'.format(loadEpoch))

                pattern = re.compile(r'-(\d+)$')
                match = pattern.search(checkpointFilename)
                if match:
                    dinoEpoch = int(match.group(1))
                
                # Sum all model weights in self.dinoTeacher
                sumWeights = tf.reduce_sum([tf.reduce_sum(variable) for variable in self.dinoTeacher.trainable_variables])
                print ('Sum of dino weights: {}'.format(sumWeights))

                status = checkpoint.restore(checkpointFilename)
                status.expect_partial()
                status.assert_existing_objects_matched()
                sumWeights = tf.reduce_sum([tf.reduce_sum(variable) for variable in self.dinoTeacher.trainable_variables])
                print ('Sum of dino weights after loading: {}'.format(sumWeights))
                print ('Loaded dino checkpoint: {}'.format(dinoEpoch))
            else:
                raise Exception("No checkpoint found")
            return dinoEpoch
        else:
            # Checkpoint
            checkpoint = tf.train.Checkpoint(logName = self.logName, teacherCenter=self.model.center, optimizer = self.model.optimizer, studentModel = self.model.student_model, teacherModel = self.model.teacher_model)
            self.checkpointManager = tf.train.CheckpointManager(checkpoint, self.checkpointDir, max_to_keep=2)

            if self.checkpointManager.latest_checkpoint:
                if loadEpoch is None:
                    checkpointFilename = self.checkpointManager.latest_checkpoint
                else:
                    checkpointFilename = os.path.join(self.checkpointDir, 'ckpt-{}'.format(loadEpoch))
                # Check using the name of the checkpoint if it already was finished
                pattern = re.compile(r'-(\d+)$')
                match = pattern.search(checkpointFilename)
                if match:
                    savedEpoch = int(match.group(1))

                if restoreTraining:
                    if savedEpoch <= self.dinoSettings['freeze_last_layer']:
                        print("Freezing last layer")
                        self.model.student_model.head.last_layer.trainable = False
                        self.model.compile(optimizer=self.optimizer)
                    print("Fitting model on fake data")
                    nBatches = 1
                    fakeGenerator = utils.datasets.Generators.FakeDinoGenerator(self.batchSize, self.nChannels,self.teacherGlobalSize,self.studentLocalSize,2, self.nLocalCrops, nBatches)
                    self.model.fit(fakeGenerator, batch_size=self.batchSize, epochs=1,verbose=1)

                print("Restore checkpoint: {}".format(checkpointFilename))
                # check if file exists
                if os.path.exists(checkpointFilename):
                    print("Checkpoint exists")

                status = checkpoint.restore(checkpointFilename)
                print("Restored checkpoint")

                #if dummyDataset is not None: 
                if restoreTraining:
                    status.assert_consumed()         

                startEpoch = savedEpoch
                print ('Loaded checkpoint {}'.format(startEpoch))
            else:
                startEpoch = 0
                print("No checkpoint found")
            return startEpoch
        
    def loadTeacher(self, loadEpoch=None):
        # Create model
        hiddenSize, teacher = self.load_base('teacher')
        
        flattenBackbone = self.dinoSettings['architecture'] == 'gan'
        reducedGanDimension = self.dinoSettings['reducedGanDimension']

        if flattenBackbone and reducedGanDimension is None:
            hidden_dim = 512
        else:
            hidden_dim = 2048
        
        if reducedGanDimension is not None:
            encoderOutputSize = (4,16,1024)
            encoderOutputDimension = encoderOutputSize[0] * encoderOutputSize[1]*encoderOutputSize[2]
            if (encoderOutputDimension) != hiddenSize:
                raise Exception("Hidden size does not match hard coded gan encoder output size")
            reducedGanFilters = round((reducedGanDimension/encoderOutputDimension)*encoderOutputSize[-1])
            hiddenSize = encoderOutputSize[0] * encoderOutputSize[1]*reducedGanFilters
        else:
            reducedGanFilters = None

        # teacher has always norm last layer
        teacherHead = DinoHead(hiddenSize,self.outputDim,use_bn = self.dinoSettings['use_bn_in_head'], hidden_dim=hidden_dim)
        self.dinoTeacher = MultiCropWrapper(backbone=teacher, head=teacherHead, hiddenSize=hiddenSize, patchSize=self.patchSize,flattenBackbone=flattenBackbone, reducedGanFilters=reducedGanFilters)
        
        # Load weights of teacher model only
        loadedEpoch = self.loadCheckpoint(restoreTraining=False, restoreTeacher=True, loadEpoch=loadEpoch)

        return self.dinoTeacher, loadedEpoch

    def loadEncoderWeights(self, loadEpoch=None, loadFlatteningLayer = False):
        self.dinoTeacher, loadedEpoch = self.loadTeacher(loadEpoch)
        encoder = self.dinoTeacher.backbone
        encoderWeights = encoder.get_weights()
        
        weights = encoderWeights
        return weights, loadedEpoch

    def loadCallbacks(self, checkpointSaver = True, tensorboard=True, attentionMapGenerator = None, embeddingEpochGenerator = None, supervisedValGenerator = None, logHistogramsEachEpoch = False):
        # Callbacks
        callbacks = []
        if checkpointSaver:
            callbacks.append(utils.models.Callbacks.CheckpointSaver(self.checkpointManager))
        
        if tensorboard:
            if self.tensorboardLocation is None:
                raise ValueError("Tensorboard callback requested, but logDir is not set")
            tensorboardDir = os.path.join(self.tensorboardLocation, self.logName.numpy().decode('utf-8'))
            if logHistogramsEachEpoch:
                histogramFreq = 1
            else:
                histogramFreq = 0
            callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=tensorboardDir, histogram_freq=histogramFreq,write_graph=False))
            #callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=tensorboardDir, histogram_freq=1,profile_batch='10,50'))

        if attentionMapGenerator is not None:
            callbacks.append(utils.models.Callbacks.AttentionMapPlotter(self.model.teacher_model,attentionMapGenerator,self.plotLocation))

        if embeddingEpochGenerator is not None:
            callbacks.append(utils.models.Callbacks.Evaluator(self.model.teacher_model, embeddingEpochGenerator, self.modelDir))
    
        if supervisedValGenerator is not None:
            testTrain = supervisedValGenerator.getClassTrainValSamples(subset='train')
            testVal = supervisedValGenerator.getClassTrainValSamples(subset='val')
            callbacks.append(utils.models.Callbacks.SupervisedEvaluator(self.model.teacher_model, testTrain, testVal, self.logLocation))

        terminateNan = utils.models.Callbacks.TerminateOnNaN()
        callbacks.append(terminateNan)

        return callbacks
    
    def summary(self):
        print()
        print("Student backbone:")
        self.model.student_model.backbone.summary()
        print()
        print("Teacher backbone:")
        self.model.teacher_model.backbone.summary()

        print()
        print("Student Head:")
        self.model.student_model.head.summary()
        print()
        print("Teacher Head:")
        self.model.teacher_model.head.summary()

        print()
        print()
        self.model.summary()
    
    def getModel(self):
        return self.model