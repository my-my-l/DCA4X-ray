import os
import sys
import time
class TextLogger(object):
    def __init__(self, filename, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.terminal.close()
        self.log.close()
        
class CompleteLogger:
    def __init__(self, root, phase='train', resume=False, resume_path = ''):
        self.root = root
        self.phase = phase
        self.visualize_directory = os.path.join(self.root, "visualize")
        self.checkpoint_directory = os.path.join(self.root, "checkpoints")
        self.epoch = 0


        os.makedirs(self.root, exist_ok=True)
        os.makedirs(self.visualize_directory, exist_ok=True)
        os.makedirs(self.checkpoint_directory, exist_ok=True)

   
        now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
        if resume:
            log_filename = os.path.join(self.root, resume_path)
            self.log_filename = log_filename
            self.logger = TextLogger(log_filename)

        else:
            log_filename = os.path.join(self.root, "{}-{}.txt".format(phase, now))
            self.log_filename = log_filename
            if os.path.exists(log_filename):
                os.remove(log_filename)
            self.logger = TextLogger(log_filename)
        
        sys.stdout = self.logger
        sys.stderr = self.logger
        if phase != 'train':
            self.set_epoch(phase)

    def set_epoch(self, epoch):
        os.makedirs(os.path.join(self.visualize_directory, str(epoch)), exist_ok=True)
        self.epoch = epoch
    

    def _get_phase_or_epoch(self):
        if self.phase == 'train':
            return str(self.epoch)
        else:
            return self.phase

    def get_image_path(self, filename: str):
        return os.path.join(self.visualize_directory, self._get_phase_or_epoch(), filename)

    def get_checkpoint_path(self, name=None):
        if name is None:
            name = self._get_phase_or_epoch()
        name = str(name)
        return os.path.join(self.checkpoint_directory, name + '_' + self.log_filename.replace('txt','pth').split('/')[-1])
    
    def get_checkpoint_root(self):
        return self.checkpoint_directory

    def close(self):
        self.logger.close()

    def get_visualize_path(self):
        return os.path.join(self.visualize_directory, self.log_filename.replace('txt','png').split('/')[-1])
