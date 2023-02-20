import os
import math
import glob
import numpy as np
from time import time
import tensorflow as tf

        
def clip_by_value_10(var,grad):
    grad = tf.where(tf.math.is_finite(grad), grad, tf.zeros_like(grad))
    return tf.clip_by_value(grad, -10.0, 10.0)

def constant_lr(epoch):
    return 1.0


class ModelEnvironment():
    def __init__(self,
                 training_data_generator,
                 validation_data_generator,
                 test_data_generator,
                 epochs,
                 eval_fns,
                 feature_generator,
                 model_settings = [],
                 summaries = None,
                 start_epoch = 0,
                 beta_1 = 0.9,
                 beta_2 = 0.999,
                 post_train_step_args = {},
                 eval_every_n_th_epoch = 1,
                 num_train_batches = None,
                 num_test_batches = None,
                 loss_names = ["total_loss"],
                 load_model = False,
                 input_keys = ["input_features","false_sample"],
                 label_keys = ["labels"],
                 gradient_processing_fn = clip_by_value_10,
                 save_dir = 'tmp'):

        if type(model_settings) == list:
            self.model_settings = model_settings
        else:
            self.model_settings = [model_settings]
        self.models = [setting['model'] for setting in self.model_settings]

        self.models_trainable = [setting['trainable'] for setting in self.model_settings]
        self.epochs = epochs
        self.feature_generator = feature_generator
        self.epoch_variable = tf.Variable(start_epoch)
        self.start_epoch = start_epoch
        self.eval_every_n_th_epoch = eval_every_n_th_epoch
        self.save_dir = save_dir
        self.load_model_flags = [setting['load_model'] for setting in self.model_settings if 'load_model' in setting.keys()]
        #Set up learning rates
        self.learning_rate_fns = [setting['learning_rate_fn'] for setting in self.model_settings if 'learning_rate_fn' in setting.keys()]
        self.base_learning_rates = [setting['base_learning_rate'] for setting in self.model_settings if 'base_learning_rate' in setting.keys()]

        self.set_up_learning_rates()

        
        #Initialize models
        init_data = [setting['init_data'] for setting in self.model_settings]
        self.initialize_models(init_data)
        self.n_vars_models = [len(model.trainable_variables) for model,trainable in zip(self.models,self.models_trainable) if trainable]
        #Set up summaries
        self.summaries = summaries
        #Set up loss function
        self.eval_fns = eval_fns(self.models,feature_generator)
        #Set up optimizers
        self.optimizer_types = [setting['optimizer_type'] for setting in self.model_settings if 'optimizer_type' in setting.keys()]

        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.loss_names = loss_names
        self.set_up_optimizers()
        self.post_train_step_args = post_train_step_args
        #Set up function to process the gradients
        self.gradient_processing_fn = gradient_processing_fn

        #Load model
        self.load_model()

        #Set up training generators
        self.training_data_generator = training_data_generator
        self.validation_data_generator = validation_data_generator
        self.test_data_generator = test_data_generator
        #Set keys
        self.input_keys = input_keys
        self.label_keys = label_keys
        
        #Get number of training batches
        if num_train_batches == None:
            try:
                self.max_number_batches = tf.data.experimental.cardinality(training_data_generator).numpy()
            except:
                self.max_number_batches = 0
        else:
            self.max_number_batches = num_train_batches

        #Get number of test batches
        if num_test_batches == None:
            try:
                self.max_number_test_batches = tf.data.experimental.cardinality(test_data_generator).numpy()
            except:
                self.max_number_test_batches = 0
        else:
            self.max_number_test_batches = num_test_batches

    def initialize_models(self,init_data):

        if len(init_data) == 1:
            for model in self.models:
                model(*init_data[0])
                try:
                    model.summary()
                except:
                    print("No summary available!")
        elif len(init_data) == len(self.models):
            for data,model in zip(init_data,self.models):
                model(*data)
                try:
                    model.summary()
                except:
                    print("No summary available!")
        else:
            raise(ValueError("Wrong data for model initialization!"))
        
    def set_up_learning_rates(self):
        self.learning_rates = []
        if len(self.learning_rate_fns) == len(self.base_learning_rates):
            for i in range(len(self.base_learning_rates)):
                #if self.models_trainable[i]:
                self.learning_rates.append(tf.Variable(
                        self.base_learning_rates[i]*self.learning_rate_fns[i](self.start_epoch),trainable=False))
        elif len(self.learning_rate_fns) == 1:
            for i in range(len(self.base_learning_rates)):
                #if self.models_trainable[i]:
                self.learning_rates.append(tf.Variable(
                        self.base_learning_rates[i]*self.learning_rate_fns[0](self.start_epoch),trainable=False))
        elif len(self.base_learning_rates) == 1:
            for i in range(len(self.learning_rate_fns)):
                #if self.models_trainable[i]:
                self.learning_rates.append(tf.Variable(
                        self.base_learning_rates[0]*self.learning_rate_fns[i](self.start_epoch),trainable=False))
        else:
            raise(NotImplementedError)


    def set_up_optimizers(self):
        self.optimizers = []
        for opt_type,lr,trainable in zip(self.optimizer_types,self.learning_rates,self.models_trainable):
            if trainable:
                if opt_type == tf.keras.optimizers.SGD:
                    self.optimizers.append(opt_type(lr,0.9,True))
                elif opt_type == tf.keras.optimizers.Adam:
                    self.optimizers.append(opt_type(lr,beta_1=self.beta_1, beta_2=self.beta_2, epsilon=1e-08))
                else:
                    self.optimizers.append(opt_type(lr))


    def get_latest_model_file(self,model_type):
        files = [os.path.join(self.save_dir, f) for f in os.listdir(self.save_dir) if
                    os.path.isfile(os.path.join(self.save_dir, f)) and "h5" in f and (model_type in f or "model" in f)]
        
        files.sort(key=lambda x: os.path.getmtime(x))
        latest = files[-1]
        return latest

    def get_model(self,model_type,print_model_name=True):
        """If self.start_epoch is 0 the loader loads the latest file
         from the directory, otherwise the file from the specified epoch is loaded"""
        latest = self.get_latest_model_file(model_type)
        split = latest.split("_")
        if self.start_epoch == 0:
            if split[-1] == ".h5":
                str_epoch = split[-2]
            else:
                str_epoch = split[-1].split(".")[0]
            self.start_epoch = int(str_epoch)
            model_to_load = latest
        else:
            split[-2] = str(self.start_epoch)
            model_to_load = latest#"_".join(split)

        if print_model_name:
            print("____________________________________")
            print("Loading model: "+latest)
            print("____________________________________")
            
        return model_to_load

    def get_checkpoint(self, model_type, print_model_name=True):
        """If self.start_epoch is 0 the loader loads the latest file
         from the directory, otherwise the file from the specified epoch is loaded"""
        latest = self.get_latest_model_file(model_type)
        latest = latest.split(".")[:-1]
        latest = ".".join(latest)
        if print_model_name:
            print("____________________________________")
            print("Loading model: " + latest)
            print("____________________________________")

        return latest
            
    def load_model(self):
        """Loads weights for all models"""
        for model,load_model_flag in zip(self.models,self.load_model_flags):

            if load_model_flag:
                #try:
                model.load_weights(self.get_model(model.model_name))
                print("Loaded model:"+model.model_name)
                #except:
                #    checkpoint = tf.train.Checkpoint(module=model)
                #    checkpoint.restore(self.get_checkpoint(model.model_name))
                #    print("Loaded model:"+model.model_name)#print("Failed to load "+str(model.model_name))

    def save_model(self,epoch):
        for trainable,model in zip(self.models_trainable,self.models):
            if trainable:
                try:
                    model.save_weights(os.path.join(self.save_dir, model.model_name+"_" + str(epoch) + "_.h5"))
                    print("Saved model to: "+os.path.join(self.save_dir, model.model_name+"_" + str(epoch) + "_.h5"))
                except:
                    try:
                        checkpoint = tf.train.Checkpoint(module=model)
                        checkpoint.save(os.path.join(self.save_dir, model.model_name+"_" + str(epoch) + "_.h5"))
                    except:
                        print("Could not save model!")
            
    def update_learning_rates(self,epoch):
        for i in range(len(self.learning_rates)):
            #if self.models_trainable[i]:
            if len(self.base_learning_rates) == 1:
                if len(self.learning_rate_fns) == 1:
                    self.learning_rates[i].assign(self.base_learning_rates[0]*self.learning_rate_fns[0](epoch))
                else:
                    self.learning_rates[i].assign(self.base_learning_rates[0]*self.learning_rate_fns[i](epoch))
            else:
                if len(self.learning_rate_fns) == 1:
                    self.learning_rates[i].assign(self.base_learning_rates[i]*self.learning_rate_fns[0](epoch))
                else:
                    self.learning_rates[i].assign(self.base_learning_rates[i]*self.learning_rate_fns[i](epoch))
        if self.summaries != None:
            self.update_learning_rate_summaries()
            self.write_learning_rate_summaries(epoch)
                    
    def predict(self,x,training = False):
        return self.eval_fns.predict(x, training=training)

    #@tf.function(experimental_relax_shapes=True)
    def compute_loss(self, x, y , training = True,epoch=-1):
        return self.eval_fns.compute_loss(x,y,training,epoch)
    
    def post_train_step(self):
        return self.eval_fns.post_train_step(self.post_train_step_args)

    @tf.function(experimental_relax_shapes=True)
    def compute_gradients(self, x, y,epoch,model,loss_name="total_loss"):
        # Pass through network
        with tf.GradientTape() as tape:
            losses,outputs = self.compute_loss(
                x, y, training=True,epoch=epoch)


        gradients = tape.gradient(losses[loss_name],model.trainable_variables)
        
        if self.gradient_processing_fn != None:
            out_grads = []
            for var,grad in zip(model.trainable_variables,gradients):

                if grad != None:
                    grad = self.gradient_processing_fn(var,grad)
                    out_grads.append(grad)
                else:
                    out_grads.append(None)
        else:
            out_grads = gradients

        return out_grads,losses,outputs

    
    def apply_gradients(self,model_index,gradients, variables):

        self.optimizers[model_index].apply_gradients(
            zip(gradients, variables)
        )


    def check_grads(self,gradients):
        for i in range(len(gradients)):
            if gradients[i] != None:
                gradients[i]=tf.where(tf.math.is_finite(gradients[i]),gradients[i],tf.zeros_like(gradients[i]))
        return gradients

    def train_step(self,x,y,epoch=-1):
        #Compute gradients
        #Apply gradients
        var_start = 0
        var_end = 0

        for i in range(len(self.optimizers)):
            if self.models_trainable[i]:
                gradients, losses, outputs = self.compute_gradients(x, y, epoch,self.models[i],loss_name=self.loss_names[i])
                var_end += self.n_vars_models[i]
                grads = gradients[var_start:var_end]
                self.apply_gradients(i,grads,self.models[i].trainable_variables)
                var_start += self.n_vars_models[i]

        self.post_train_step()
        
        return losses,outputs

    def update_learning_rate_summaries(self):
        lr_dict = {}
        for lr_name,lr in zip(self.summaries.learning_rate_names,self.learning_rates):
            lr_dict[lr_name] = lr
            
        self.summaries.update_lr(lr_dict)

    def update_summaries(self,losses,outputs,y=None,mode='train'):

        if self.summaries != None:
            
            scalars = losses
            #Update scalar summaries
            self.summaries.update(scalars,mode)
            #Update image summaries
            self.summaries.update_image_data(outputs,mode)
            #Update audio summaries
            self.summaries.update_audio_data(outputs,mode)
        
    def write_summaries(self,epoch,mode="train"):
        
        if self.summaries != None:
            # Write scalar summaries
            self.summaries.write(epoch,mode)
            #Write image summaries
            self.summaries.write_image_summaries(epoch,mode)
            #Write audio summaries
            self.summaries.write_audio_summaries(epoch,mode)
                
    def write_learning_rate_summaries(self,epoch):
        if self.summaries != None:
            # Write summaries
            self.summaries.write_lr(epoch)
            
    def reset_summaries(self):
        
        if self.summaries != None:
            # Write summaries
            self.summaries.reset_summaries()


    def get_data_for_keys(self,xy,keys):
        """Returns list of input data"""
        x = []
        for key in keys:
            x.append(xy[key])
            
        return x

    def predict_dataset(self,data_generator,num_batches = 1,summaries = None,use_progbar = False,epoch=0):
        if use_progbar:
            prog = tf.keras.utils.Progbar(num_batches)
        
        all_predictions = []
        batch = 0
        for xy in data_generator:
            start_time = time()
            x = self.get_data_for_keys(xy,self.input_keys)
            y = self.get_data_for_keys(xy,self.label_keys)
            losses,outputs = self.compute_loss(x,y,False)
            all_predictions.append(outputs)
            
            # Update summaries
            for mode in self.summaries.modes:
                self.update_summaries(losses,outputs,y,mode)

            if use_progbar:
                if self.summaries != None:
                    for mode in self.summaries.modes:
                        summary_list = self.summaries.get_summary_list(mode)
                else:
                    summary_list = []
                
                summary_list += [("time / step", np.round(time() - start_time, 2))]
                prog.update(batch, summary_list)
            batch += 1

        # Write train summaries
        for mode in self.summaries.modes:
            self.write_summaries(epoch+1, mode)
                                
        return all_predictions

    def train(self):
        if self.start_epoch == 0:
            print("Saving...")
            self.save_model(0)

        print("Starting training...")

        if self.summaries != None:
            summary_list = self.summaries.get_summary_list('train')
        else:
            summary_list = []

        # Progress bar
        prog = tf.keras.utils.Progbar(self.max_number_batches,stateful_metrics=[summary[0] for summary in summary_list])
        for epoch in range(self.start_epoch, self.epochs):
            # Update learning rates
            self.update_learning_rates(epoch)
            #Update epoch variable
            self.epoch_variable.assign(epoch)
            # Start epoch
            print("Starting epoch " + str(epoch + 1))
            batch = 0

            for train_xy in self.training_data_generator:
                train_x = self.get_data_for_keys(train_xy,self.input_keys)
                train_y = self.get_data_for_keys(train_xy,self.label_keys)
                start_time = time()
                losses,outputs = self.train_step(train_x, train_y)

                # Update summaries
                self.update_summaries(losses,outputs,train_y,'train')

                batch += 1

                if self.summaries != None:
                    summary_list = self.summaries.get_summary_list('train')
                else:
                    summary_list = []
                summary_list += [("time / step", np.round(time() - start_time, 2))]
                prog.update(batch, summary_list)

                if batch >= self.max_number_batches:
                    break

            # Write train summaries
            self.write_summaries(epoch+1, 'train')
            self.eval_fns.reset()

            if (epoch % self.eval_every_n_th_epoch == 0 or epoch == self.epochs - 1):
                if (self.eval_every_n_th_epoch >1 and epoch>0) or self.eval_every_n_th_epoch==1:
                    check_epoch = True
                else:
                    check_epoch = False
            else:
                check_epoch = False

            if self.validation_data_generator != None and check_epoch:
                batch = 0
                for validation_xy in self.validation_data_generator:
                    validation_x = self.get_data_for_keys(validation_xy,self.input_keys)
                    validation_y = self.get_data_for_keys(validation_xy,self.label_keys)
                    losses,outputs = self.compute_loss(validation_x,
                                                                validation_y,
                                                                training=False)

                    # Update summaries
                    self.update_summaries(losses,outputs,validation_y,"val")
                    batch += 1

                # Write validation summaries
                self.write_summaries(epoch+1,"val")
                self.eval_fns.reset()

            if self.test_data_generator != None and check_epoch:
                batch = 0
                for test_xy in self.test_data_generator:
                    test_x = self.get_data_for_keys(test_xy,self.input_keys)
                    test_y = self.get_data_for_keys(test_xy,self.label_keys)

                    losses,outputs = self.compute_loss(test_x,
                                                                test_y,
                                                                training=False)

                    # Update summaries
                    self.update_summaries(losses,outputs,test_y,"test")
                    batch += 1
                # Write test summaries
                self.write_summaries(epoch+1,"test")
                self.eval_fns.reset()

            if self.summaries != None and (epoch % self.eval_every_n_th_epoch == 0 or epoch == self.epochs-1):
                summary_list = self.summaries.get_summary_list('eval')
                scalars = [str(x[0])+":"+str(x[1].numpy()) for x in summary_list]
                print(",".join(scalars))

            #Reset summaries after epoch
            if self.summaries != None:
                self.summaries.reset_summaries()
            #Save the model weights
            self.save_model(epoch+1)




