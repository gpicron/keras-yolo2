from typing import NamedTuple

import sys


from keras.models import Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers.advanced_activations import LeakyReLU

import numpy as np
import os
import cv2


from utils import decode_netout, compute_overlap, compute_ap
from keras.applications.mobilenet import MobileNet
from keras.layers.merge import concatenate
from keras.optimizers import SGD, Adam, RMSprop
from preprocessing import BatchGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from backend import TinyYoloFeature, FullYoloFeature, MobileNetFeature, MobileNetV2Feature, SqueezeNetFeature, Inception3Feature, VGG16Feature, ResNet50Feature

import keras.backend as K
import tensorflow as tf


class YOLO(object):
    def __init__(self, backend,
                       input_size, 
                       labels, 
                       max_box_per_image,
                       anchors):

        self.input_size = input_size
        
        self.labels   = list(labels)
        self.nb_class = len(self.labels)
        self.nb_box   = len(anchors)//2
        self.class_wt = np.ones(self.nb_class, dtype='float32')
        self.anchors  = anchors

        self.max_box_per_image = max_box_per_image

        ##########################
        # Make the model
        ##########################

        # make the feature extractor layers
        input_image     = Input(shape=(self.input_size, self.input_size, 3), dtype= "float32")
        self.true_boxes = Input(shape=(1, 1, 1, max_box_per_image , 4))  

        if backend == 'Inception3':
            self.feature_extractor = Inception3Feature(self.input_size)  
        elif backend == 'SqueezeNet':
            self.feature_extractor = SqueezeNetFeature(self.input_size)        
        elif backend == 'MobileNet':
            self.feature_extractor = MobileNetFeature(self.input_size)
        elif backend == 'MobileNetV2':
            self.feature_extractor = MobileNetV2Feature(self.input_size)
        elif backend == 'Full Yolo':
            self.feature_extractor = FullYoloFeature(self.input_size)
        elif backend == 'Tiny Yolo':
            self.feature_extractor = TinyYoloFeature(self.input_size)
        elif backend == 'VGG16':
            self.feature_extractor = VGG16Feature(self.input_size)
        elif backend == 'ResNet50':
            self.feature_extractor = ResNet50Feature(self.input_size)
        else:
            raise Exception('Architecture not supported! Only support Full Yolo, Tiny Yolo, MobileNet, SqueezeNet, VGG16, ResNet50, and Inception3 at the moment!')

        print(self.feature_extractor.get_output_shape())    
        self.grid_h, self.grid_w = self.feature_extractor.get_output_shape()        
        #features = self.feature_extractor.extract(input_image)
        features = self.feature_extractor.feature_extractor.output

        # make the object detection layer
        output = Conv2D(self.nb_box * (4 + 1 + self.nb_class), 
                        (1,1), strides=(1,1), 
                        padding='same', 
                        name='DetectionLayer', 
                        kernel_initializer='lecun_normal')(features)
        output = Reshape((self.grid_h, self.grid_w, self.nb_box, 4 + 1 + self.nb_class))(output)
        self.true_output = output
        output = Lambda(lambda args: args[0])([output, self.true_boxes])

        #self.model = Model([input_image, self.true_boxes], output)
        self.model = Model([self.feature_extractor.feature_extractor.input , self.true_boxes], output)
        self.tflite_model = None
        self.lite_interpreter = None

        self.frozen_graph = None

        self.open_vino_exec_net = None

        self.predict = self.predict_keras


        # initialize the weights of the detection layer
        layer = self.model.layers[-4]
        weights = layer.get_weights()

        new_kernel = np.random.normal(size=weights[0].shape)/(self.grid_h*self.grid_w)
        new_bias   = np.random.normal(size=weights[1].shape)/(self.grid_h*self.grid_w)

        layer.set_weights([new_kernel, new_bias])

        # print a summary of the whole model
        self.model.summary()


    def custom_loss(self, y_true, y_pred):
        mask_shape = tf.shape(y_true)[:4]
        
        cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(self.grid_w), [self.grid_h]), (1, self.grid_h, self.grid_w, 1, 1)))
        cell_y = tf.transpose(cell_x, (0,2,1,3,4))

        cell_grid = tf.tile(tf.concat([cell_x,cell_y], -1), [self.batch_size, 1, 1, self.nb_box, 1])
        
        coord_mask = tf.zeros(mask_shape)
        conf_mask  = tf.zeros(mask_shape)
        class_mask = tf.zeros(mask_shape)
        
        seen = tf.Variable(0.)
        total_recall = tf.Variable(0.)
        
        """
        Adjust prediction
        """
        ### adjust x and y      
        pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid
        
        ### adjust w and h
        pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(self.anchors, [1,1,1,self.nb_box,2])
        
        ### adjust confidence
        pred_box_conf = tf.sigmoid(y_pred[..., 4])
        
        ### adjust class probabilities
        pred_box_class = y_pred[..., 5:]
        
        """
        Adjust ground truth
        """
        ### adjust x and y
        true_box_xy = y_true[..., 0:2] # relative position to the containing cell
        
        ### adjust w and h
        true_box_wh = y_true[..., 2:4] # number of cells accross, horizontally and vertically
        
        ### adjust confidence
        true_wh_half = true_box_wh / 2.
        true_mins    = true_box_xy - true_wh_half
        true_maxes   = true_box_xy + true_wh_half
        
        pred_wh_half = pred_box_wh / 2.
        pred_mins    = pred_box_xy - pred_wh_half
        pred_maxes   = pred_box_xy + pred_wh_half       
        
        intersect_mins  = tf.maximum(pred_mins,  true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
        
        true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
        pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores  = tf.truediv(intersect_areas, union_areas)
        
        true_box_conf = iou_scores * y_true[..., 4]
        
        ### adjust class probabilities
        true_box_class = tf.argmax(y_true[..., 5:], -1)
        
        """
        Determine the masks
        """
        ### coordinate mask: simply the position of the ground truth boxes (the predictors)
        coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * self.coord_scale
        
        ### confidence mask: penelize predictors + penalize boxes with low IOU
        # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
        true_xy = self.true_boxes[..., 0:2]
        true_wh = self.true_boxes[..., 2:4]
        
        true_wh_half = true_wh / 2.
        true_mins    = true_xy - true_wh_half
        true_maxes   = true_xy + true_wh_half
        
        pred_xy = tf.expand_dims(pred_box_xy, 4)
        pred_wh = tf.expand_dims(pred_box_wh, 4)
        
        pred_wh_half = pred_wh / 2.
        pred_mins    = pred_xy - pred_wh_half
        pred_maxes   = pred_xy + pred_wh_half    
        
        intersect_mins  = tf.maximum(pred_mins,  true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
        
        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores  = tf.truediv(intersect_areas, union_areas)

        best_ious = tf.reduce_max(iou_scores, axis=4)
        conf_mask = conf_mask + tf.to_float(best_ious < 0.6) * (1 - y_true[..., 4]) * self.no_object_scale
        
        # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
        conf_mask = conf_mask + y_true[..., 4] * self.object_scale
        
        ### class mask: simply the position of the ground truth boxes (the predictors)
        class_mask = y_true[..., 4] * tf.gather(self.class_wt, true_box_class) * self.class_scale       
        
        """
        Warm-up training
        """
        no_boxes_mask = tf.to_float(coord_mask < self.coord_scale/2.)
        seen = tf.assign_add(seen, 1.)
        
        true_box_xy, true_box_wh, coord_mask = tf.cond(tf.less(seen, self.warmup_batches+1), 
                              lambda: [true_box_xy + (0.5 + cell_grid) * no_boxes_mask, 
                                       true_box_wh + tf.ones_like(true_box_wh) * \
                                       np.reshape(self.anchors, [1,1,1,self.nb_box,2]) * \
                                       no_boxes_mask, 
                                       tf.ones_like(coord_mask)],
                              lambda: [true_box_xy, 
                                       true_box_wh,
                                       coord_mask])
        
        """
        Finalize the loss
        """
        nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
        nb_conf_box  = tf.reduce_sum(tf.to_float(conf_mask  > 0.0))
        nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))
        
        loss_xy    = tf.reduce_sum(tf.square(true_box_xy-pred_box_xy)     * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_wh    = tf.reduce_sum(tf.square(true_box_wh-pred_box_wh)     * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_conf  = tf.reduce_sum(tf.square(true_box_conf-pred_box_conf) * conf_mask)  / (nb_conf_box  + 1e-6) / 2.
        loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
        loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)
        
        loss = tf.cond(tf.less(seen, self.warmup_batches+1), 
                      lambda: loss_xy + loss_wh + loss_conf + loss_class + 10,
                      lambda: loss_xy + loss_wh + loss_conf + loss_class)
        
        if self.debug:
            nb_true_box = tf.reduce_sum(y_true[..., 4])
            nb_pred_box = tf.reduce_sum(tf.to_float(true_box_conf > 0.5) * tf.to_float(pred_box_conf > 0.3))
            
            current_recall = nb_pred_box/(nb_true_box + 1e-6)
            total_recall = tf.assign_add(total_recall, current_recall) 

            loss = tf.Print(loss, [loss_xy], message='Loss XY \t', summarize=1000)
            loss = tf.Print(loss, [loss_wh], message='Loss WH \t', summarize=1000)
            loss = tf.Print(loss, [loss_conf], message='Loss Conf \t', summarize=1000)
            loss = tf.Print(loss, [loss_class], message='Loss Class \t', summarize=1000)
            loss = tf.Print(loss, [loss], message='Total Loss \t', summarize=1000)
            loss = tf.Print(loss, [current_recall], message='Current Recall \t', summarize=1000)
            loss = tf.Print(loss, [total_recall/seen], message='Average Recall \t', summarize=1000)
        
        return loss

    def load_weights(self, weight_path):
        self.model.load_weights(weight_path, by_name=True, skip_mismatch=True)

    def convert_to_lite(self):

        model_path = "tmp_model.h5"
        infer_model = Model([self.model.inputs[0]], [self.true_output])

        infer_model.save(model_path)
        converter = tf.contrib.lite.TFLiteConverter.from_keras_model_file(model_path)

        self.tflite_model = converter.convert()
        #open("%s.tflite" % weight_path, "wb").write(self.tflite_model)

        self.lite_interpreter = tf.contrib.lite.Interpreter(model_content=self.tflite_model)
        self.lite_interpreter.allocate_tensors()

    def convert_to_frozen_tf(self):
        from tensorflow.python.framework import graph_io

        model_fname = "tmp_model.h5"
        infer_model = Model([self.model.inputs[0]], [self.true_output])
        infer_model.save(model_fname)
        del infer_model

        # Clear any previous session.
        tf.keras.backend.clear_session()
        K.clear_session()
        tf.keras.backend.set_learning_phase(0)
        K.set_learning_phase(0)


        def freeze_graph(graph, session, output, save_pb_dir='.', save_pb_name='tmp_frozen_model.pb', save_pb_as_text=False):
            with graph.as_default():
                graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
                graphdef_frozen = tf.graph_util.convert_variables_to_constants(session, graphdef_inf, output)
                graph_io.write_graph(graphdef_frozen, save_pb_dir, save_pb_name, as_text=save_pb_as_text)

                return graphdef_frozen


        # This line must be executed before loading Keras model.

        model = tf.keras.models.load_model(model_fname)

        session = tf.keras.backend.get_session()

        self.INPUT_NODE = [t.op.name for t in model.inputs]
        self.OUTPUT_NODE = [t.op.name for t in model.outputs]
        print("in/out layer names:", self.INPUT_NODE, self.OUTPUT_NODE)

        self.frozen_graph_def = freeze_graph(session.graph, session, [out.op.name for out in model.outputs])


        self.frozen_graph = tf.Graph()

        with self.frozen_graph.as_default():
            tf.import_graph_def(self.frozen_graph_def)

            self.frozen_net_inp = self.frozen_graph.get_tensor_by_name('import/' + self.INPUT_NODE[0] + ":0")
            self.frozen_net_out = self.frozen_graph.get_tensor_by_name('import/' + self.OUTPUT_NODE[0]  + ":0")

            self.frozen_session = tf.Session(graph=self.frozen_graph)

    def convert_to_openvino(self, plugin="CPU"):
        self.convert_to_frozen_tf()

        try:
            sys.path.append('/opt/intel/computer_vision_sdk/deployment_tools/model_optimizer')

            from mo.utils.versions_checker import check_python_version
            ret_code = check_python_version()
            if ret_code:
                sys.exit(ret_code)

            from mo.main import driver
            from mo.utils.cli_parser import get_tf_cli_parser

            args = get_tf_cli_parser().parse_args({})
            args.framework = 'tf'
            args.input_model = 'tmp_frozen_model.pb'
            args.input_shape = '[1,%s,%s,3]' % (self.input_size,self.input_size)

            ret_code = driver(args)
            if ret_code:
                sys.exit(ret_code)

            from openvino import inference_engine as ie
            from openvino.inference_engine import IENetwork, IEPlugin
        except Exception as e:
            exception_type = type(e).__name__
            print("The following error happened while importing Python API module:\n[ {} ] {}".format(exception_type, e))
            sys.exit(1)

        # Plugin initialization for specified device and load extensions library if specified.
        plugin_dir = None
        model_xml = './tmp_frozen_model.xml'
        model_bin = './tmp_frozen_model.bin'
        # Devices: GPU (intel), CPU, MYRIAD
        plugin = IEPlugin(plugin, plugin_dirs=plugin_dir)
        # Read IR
        net = IENetwork(model=model_xml, weights=model_bin)
        assert len(net.inputs.keys()) == 1
        assert len(net.outputs) == 1
        self.ov_input_blob = next(iter(net.inputs))
        self.ov_out_blob = next(iter(net.outputs))
        # Load network to the plugin
        self.open_vino_exec_net = plugin.load(network=net)
        del net



    def train(self, train_imgs,     # the list of images to train the model
                    valid_imgs,     # the list of images used to validate the model
                    train_times,    # the number of time to repeat the training set, often used for small datasets
                    valid_times,    # the number of times to repeat the validation set, often used for small datasets
                    nb_epochs,      # number of epoches
                    learning_rate,  # the learning rate
                    batch_size,     # the size of the batch
                    warmup_epochs,  # number of initial batches to let the model familiarize with the new dataset
                    object_scale,
                    no_object_scale,
                    coord_scale,
                    class_scale,
                    saved_weights_name='best_weights.h5',
                    debug=False,
                    initial_epoch=0,
                    nb_epoch_freeze_tower=0):

        self.batch_size = batch_size

        self.object_scale    = object_scale
        self.no_object_scale = no_object_scale
        self.coord_scale     = coord_scale
        self.class_scale     = class_scale

        self.debug = debug

        ############################################
        # Make train and validation generators
        ############################################

        generator_config = {
            'IMAGE_H'         : self.input_size, 
            'IMAGE_W'         : self.input_size,
            'GRID_H'          : self.grid_h,  
            'GRID_W'          : self.grid_w,
            'BOX'             : self.nb_box,
            'LABELS'          : self.labels,
            'CLASS'           : len(self.labels),
            'ANCHORS'         : self.anchors,
            'BATCH_SIZE'      : self.batch_size,
            'TRUE_BOX_BUFFER' : self.max_box_per_image,
        }    

        train_generator = BatchGenerator(train_imgs, 
                                     generator_config, 
                                     norm=self.feature_extractor.normalize)
        valid_generator = BatchGenerator(valid_imgs, 
                                     generator_config, 
                                     norm=self.feature_extractor.normalize,
                                     jitter=False)   
                                     
        self.warmup_batches  = warmup_epochs * (train_times*len(train_generator) + valid_times*len(valid_generator))   

        ############################################
        # Make a few callbacks
        ############################################

        early_stop = EarlyStopping(monitor='val_loss', 
                           min_delta=0.001, 
                           patience=3, 
                           mode='min', 
                           verbose=1)
        checkpoint = ModelCheckpoint(saved_weights_name, 
                                     monitor='val_loss', 
                                     verbose=1, 
                                     save_best_only=True, 
                                     mode='min', 
                                     period=1)
        tensorboard = TensorBoard(log_dir=os.path.expanduser('~/logs/'),
                                  histogram_freq=0, 
                                  #write_batch_performance=True,
                                  write_graph=True, 
                                  write_images=False)

        ############################################
        # Start the training process
        ############################################        
        optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        total_epochs = warmup_epochs + nb_epochs

        frozen_epochs = min(nb_epoch_freeze_tower, total_epochs)
        unfrozen_epochs = total_epochs - frozen_epochs

        if frozen_epochs > 0:
            self.feature_extractor.freeze()

            self.model.compile(loss=self.custom_loss, optimizer=optimizer)

            self.model.fit_generator(generator        = train_generator,
                                     steps_per_epoch  = len(train_generator) * train_times,
                                     epochs           = frozen_epochs,
                                     verbose          = 2 if debug else 1,
                                     validation_data  = valid_generator,
                                     validation_steps = len(valid_generator) * valid_times,
                                     callbacks        = [early_stop, checkpoint, tensorboard],
                                     workers          = 3,
                                     max_queue_size   = 8,
                                     initial_epoch    = initial_epoch)

        if unfrozen_epochs > 0:
            self.feature_extractor.unfreeze()

            self.model.compile(loss=self.custom_loss, optimizer=optimizer)

            self.model.fit_generator(generator        = train_generator,
                                     steps_per_epoch  = len(train_generator) * train_times,
                                     epochs           = unfrozen_epochs,
                                     verbose          = 2 if debug else 1,
                                     validation_data  = valid_generator,
                                     validation_steps = len(valid_generator) * valid_times,
                                     callbacks        = [early_stop, checkpoint, tensorboard],
                                     workers          = 3,
                                     max_queue_size   = 8,
                                     initial_epoch    = initial_epoch)
    
        ############################################
        # Compute mAP on the validation set
        ############################################
        average_precisions = self.evaluate(valid_generator)     

        # print evaluation
        for label, average_precision in average_precisions.items():
            print(self.labels[label], '{:.4f}'.format(average_precision))
        print('mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions)))         

    def evaluate(self, 
                 generator, 
                 iou_threshold=0.3,
                 score_threshold=0.3,
                 max_detections=100,
                 save_path=None):
        """ Evaluate a given dataset using a given model.
        code originally from https://github.com/fizyr/keras-retinanet

        # Arguments
            generator       : The generator that represents the dataset to evaluate.
            model           : The model to evaluate.
            iou_threshold   : The threshold used to consider when a detection is positive or negative.
            score_threshold : The score confidence threshold to use for detections.
            max_detections  : The maximum number of detections to use per image.
            save_path       : The path to save images with visualized detections to.
        # Returns
            A dict mapping class names to mAP scores.
        """    
        # gather all detections and annotations
        all_detections     = [[None for i in range(generator.num_classes())] for j in range(generator.size())]
        all_annotations    = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

        for i in range(generator.size()):
            raw_image = generator.load_image(i)
            raw_height, raw_width, raw_channels = raw_image.shape

            # make the boxes and the labels
            pred_boxes  = self.predict(raw_image)

            
            score = np.array([box.score for box in pred_boxes])
            pred_labels = np.array([box.label for box in pred_boxes])        
            
            if len(pred_boxes) > 0:
                pred_boxes = np.array([[box.xmin*raw_width, box.ymin*raw_height, box.xmax*raw_width, box.ymax*raw_height, box.score] for box in pred_boxes])
            else:
                pred_boxes = np.array([[]])  
            
            # sort the boxes and the labels according to scores
            score_sort = np.argsort(-score)
            pred_labels = pred_labels[score_sort]
            pred_boxes  = pred_boxes[score_sort]
            
            # copy detections to all_detections
            for label in range(generator.num_classes()):
                all_detections[i][label] = pred_boxes[pred_labels == label, :]
                
            annotations = generator.load_annotation(i)
            
            # copy detections to all_annotations
            for label in range(generator.num_classes()):
                all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()
                
        # compute mAP by comparing all detections and all annotations
        average_precisions = {}
        
        for label in range(generator.num_classes()):
            false_positives = np.zeros((0,))
            true_positives  = np.zeros((0,))
            scores          = np.zeros((0,))
            num_annotations = 0.0

            for i in range(generator.size()):
                detections           = all_detections[i][label]
                annotations          = all_annotations[i][label]
                num_annotations     += annotations.shape[0]
                detected_annotations = []

                for d in detections:
                    scores = np.append(scores, d[4])

                    if annotations.shape[0] == 0:
                        false_positives = np.append(false_positives, 1)
                        true_positives  = np.append(true_positives, 0)
                        continue

                    overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                    assigned_annotation = np.argmax(overlaps, axis=1)
                    max_overlap         = overlaps[0, assigned_annotation]

                    if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                        false_positives = np.append(false_positives, 0)
                        true_positives  = np.append(true_positives, 1)
                        detected_annotations.append(assigned_annotation)
                    else:
                        false_positives = np.append(false_positives, 1)
                        true_positives  = np.append(true_positives, 0)

            # no annotations -> AP for this class is 0 (is this correct?)
            if num_annotations == 0:
                average_precisions[label] = 0
                continue

            # sort by score
            indices         = np.argsort(-scores)
            false_positives = false_positives[indices]
            true_positives  = true_positives[indices]

            # compute false positives and true positives
            false_positives = np.cumsum(false_positives)
            true_positives  = np.cumsum(true_positives)

            # compute recall and precision
            recall    = true_positives / num_annotations
            precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

            # compute average precision
            average_precision  = compute_ap(recall, precision)  
            average_precisions[label] = average_precision

        return average_precisions    


    def use_inference_engine(self, inference_engine):
        if inference_engine == 'keras':
            self.predict = self.predict_keras
        elif inference_engine == 'tflite':
            self.convert_to_lite()
            self.predict = self.predict_tflite
        elif inference_engine == 'tf':
            self.convert_to_frozen_tf()
            self.predict = self.predict_tffrozen
        elif inference_engine == 'openvino':
            self.convert_to_openvino()
            self.predict = self.predict_openvino
        else:
            raise Exception("Not supported")

    def predict_keras(self, image):
        image_h, image_w, _ = image.shape
        image = cv2.resize(image, (self.input_size, self.input_size))
        image = self.feature_extractor.normalize(image)

        input_image = image[:,:,::-1]
        input_image = np.expand_dims(input_image, 0)
        dummy_array = np.zeros((1,1,1,1,self.max_box_per_image,4))

        netout = self.model.predict([input_image, dummy_array])[0]
        boxes  = decode_netout(netout, self.anchors, self.nb_class)

        return boxes

    def predict_tffrozen(self,image):
        image_h, image_w, _ = image.shape
        image = cv2.resize(image, (self.input_size, self.input_size))
        image = self.feature_extractor.normalize(image)

        input_image = image[:,:,::-1]
        input_image = np.expand_dims(input_image, 0)

        input_tensor= input_image.astype(np.float32)
        netout = self.frozen_session.run(self.frozen_net_out, feed_dict={self.frozen_net_inp: input_tensor})

        boxes  = decode_netout(netout[0], self.anchors, self.nb_class)

        return boxes

    def predict_tflite(self,image):
        # Get input and output tensors.
        input_details = self.lite_interpreter.get_input_details()
        output_details = self.lite_interpreter.get_output_details()

        image_h, image_w, _ = image.shape
        image = cv2.resize(image, (self.input_size, self.input_size))
        image = self.feature_extractor.normalize(image)

        input_image = image[:,:,::-1]
        input_image = np.expand_dims(input_image, 0)
        input_tensor= input_image.astype(np.float32)

        self.lite_interpreter.set_tensor(input_details[0]['index'], input_tensor)

        self.lite_interpreter.invoke()

        netout = self.lite_interpreter.get_tensor(output_details[0]['index'])

        boxes  = decode_netout(netout[0], self.anchors, self.nb_class)

        return boxes

    def predict_openvino(self, image):
        image_h, image_w, _ = image.shape
        image = cv2.resize(image, (self.input_size, self.input_size))
        image = self.feature_extractor.normalize(image)

        input_image = image[:,:,::-1]
        input_image = input_image.transpose((2, 0, 1))
        input_image = np.expand_dims(input_image, 0)
        input_tensor= input_image.astype(np.float32)

        # Run inference
        res = self.open_vino_exec_net.infer(inputs={self.ov_input_blob: input_tensor})
        # Access the results and get the index of the highest confidence score
        output_node_name = list(res.keys())[0]
        netout = res[output_node_name]

        boxes  = decode_netout(netout[0], self.anchors, self.nb_class)

        return boxes
