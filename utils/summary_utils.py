import os
import tensorflow as tf

class Summaries(object):
    def __init__(self,scalar_summary_names,learning_rate_names=[],image_summary_settings={},audio_summary_settings={},save_dir="/tmp",modes = ["train","val","test"],summaries_to_print={}):
        
        self.scalar_summary_names = scalar_summary_names
        self.image_summary_settings = image_summary_settings
        self.audio_summary_settings = audio_summary_settings
        self.save_dir = save_dir
        self.modes = modes
        self.summaries_to_print = summaries_to_print
        self.scalar_summaries = {}
        self.image_data = {}
        self.lr_summaries = {}
        self.learning_rate_names = learning_rate_names
        self.summary_writers = {}
        self.create_summary_writers()
        self.define_scalar_summaries()
        self.define_learning_rate_summaries()
        
    def get_summary_list(self,mode = 'train'):
        
        summary_list = []
        if 'eval' in mode:
            for tmp_mode in self.modes:
                for key in self.summaries_to_print[mode]:
                    summary_list.append((tmp_mode+'_'+key,self.scalar_summaries[tmp_mode+'_'+key].result()))
        else:
            if mode in self.summaries_to_print.keys():
                for key in self.summaries_to_print[mode]:
                    summary_list.append((key,self.scalar_summaries[mode+'_'+key].result()))
        return summary_list
        
    def create_summary_writers(self):
        for mode in self.modes:
            log_dir = os.path.join(self.save_dir, 'logs',mode)        
            self.summary_writers[mode] = tf.summary.create_file_writer(log_dir)

    def define_learning_rate_summaries(self):
        for key in self.learning_rate_names:
            self.lr_summaries[key] = tf.keras.metrics.Mean(key, dtype=tf.float32)

    def define_scalar_summaries(self):
        for mode in self.modes:
            for key in self.scalar_summary_names:
                self.scalar_summaries[mode+'_'+key] = tf.keras.metrics.Mean(mode+'_'+key, dtype=tf.float32)
                
        
    def update(self,scalars,mode="train"):

        for key in scalars.keys():
            #Get scalar
            scalar = scalars[key]
            if mode+"_"+key in self.scalar_summaries.keys():
                #Update summary
                self.scalar_summaries[mode+"_"+key].update_state(scalar)
            else:
                print("Summary for key: "+key+" not found!")
    
    def update_lr(self,lrs):
        for key in self.lr_summaries.keys():
            self.lr_summaries[key].update_state(lrs[key])
        
    def write(self,epoch,mode="train"):
        
        for key in self.scalar_summaries.keys():
            if mode in key.lower():
                scalar_map_key = key.split(mode+"_")[-1]
                # Write summaries
                with self.summary_writers[mode].as_default():
                    tf.summary.scalar(scalar_map_key,
                            self.scalar_summaries[key].result(), step=epoch)
                    
    def update_image_data(self,outputs,mode="train"):
        self.image_data = {}
        for key in outputs.keys():
            if mode in self.image_summary_settings.keys():
                if key in self.image_summary_settings[mode]:
                    self.image_data[key] = outputs[key]


    def update_audio_data(self,outputs,mode="train"):
        self.audio_data = {}
        for key in outputs.keys():
            if mode in self.audio_summary_settings.keys():
                if key in self.audio_summary_settings[mode]:
                    self.audio_data[key] = outputs[key]

    def write_image_summaries(self,epoch,mode="train"):
        with self.summary_writers[mode].as_default():
            if mode in self.image_summary_settings.keys():
                try:
                    n_images = self.image_summary_settings["n_images"]
                except:
                    n_images = 1
                    
                for image_summary in self.image_summary_settings[mode]:
                    if image_summary in self.image_data.keys():
                        tf.summary.image(image_summary, self.image_data[image_summary],
                                        step=epoch,
                                        max_outputs=n_images)

    def write_audio_summaries(self, epoch, mode="train"):
        with self.summary_writers[mode].as_default():
            if mode in self.audio_summary_settings.keys():
                try:
                    n_audio = self.audio_summary_settings["n_audio"]
                except:
                    n_audio = 1

                for audio_summary in self.audio_summary_settings[mode]:
                    if audio_summary in self.audio_data.keys():
                        tf.summary.audio(audio_summary,self.audio_data[audio_summary],
                                         self.audio_summary_settings["sample_rate"],
                                         step=epoch,
                                         max_outputs=n_audio)

    def write_lr(self,epoch):
        
        for key in self.lr_summaries.keys():
            # Write summaries
            with self.summary_writers['train'].as_default():
                tf.summary.scalar(key,self.lr_summaries[key].result(), step=epoch)


    def reset_summaries(self):
        for key in self.scalar_summaries.keys():
            self.scalar_summaries[key].reset_states()
        for key in self.lr_summaries.keys():
            self.lr_summaries[key].reset_states()
