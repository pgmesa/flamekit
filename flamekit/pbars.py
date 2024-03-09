
import tqdm
from flamekit.callbacks import Callback


class ProgressBar(Callback):
    
    def create_pbar(self, *args, **kwargs):
        """ Creates the progress bar. """
        raise NotImplementedError
        
    def update_pbar_metrics(self, pbar, metrics):
        """ Updates the progress bar with the given metrics. """
        raise NotImplementedError
    

class TQDMProgressBar(ProgressBar):
    
    def __init__(self, pbar_size:int=30, ascii=None, desc_above=False,
                 show_desc=True, show_elapsed_time=True, show_remaining_time=True, show_rate=True,
                 show_postfix=True, show_n_fmt=True, show_total_fmt=True, show_percentage=True,
                 pbar_frames=('|','|'), l_bar=None, r_bar=None) -> None:
        """ 
        Customizable terminal progress bar using tqdm.
        
        default tqdm l_bar = '{desc}: {percentage:3.0f}% |'
        default tqdm r_bar = '| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
        
        Args:
            pbar_size (int): Size of the progress bar.
            ascii (str): ASCII art for the progress bar.
            l_bar (str): Left bar of the progress bar.
            r_bar (str): Right bar of the progress bar.
        
        Returns:
            None.
        
        Some values for ascii parameter:
        >>> - ' >='
            - '.>='
            - ' ▖▘▝▗▚▞█'
            - '░▒█'
            - ' ▁▂▃▄▅▆▇█'
        """ 
        super().__init__()
        self.ascii = ascii
        self.l_bar = l_bar
        self.r_bar = r_bar
        self.size = pbar_size
        self.desc_above = desc_above
        self.pbar_frames = pbar_frames
        
        self.desc_str = '{desc}'
        self.show_desc = show_desc
        self.elapsed_time_str = '{elapsed}'
        self.show_elapsed_time = show_elapsed_time
        self.remaining_time_str = '{remaining}'
        self.show_remaining_time = show_remaining_time
        self.rate_str = '{rate_fmt}'
        self.show_rate = show_rate
        self.postfix_str = '{postfix}'
        self.show_postfix = show_postfix
        self.n_fmt_str = '{n_fmt}'
        self.show_n_fmt = show_n_fmt
        self.total_fmt_str = '{total_fmt}'
        self.show_total_fmt = show_total_fmt
        self.percentage_str = '{percentage:3.0f}'
        self.show_percentage = show_percentage
        
        self.l_bar = l_bar
        self.r_bar = r_bar
        
        self.pbar = None
        self.predict_pbar = None
    
    def create_pbar(self, desc=None, total=None, position=0, leave=False) -> tqdm.tqdm:
        """ Creates the progress bar. """
        pbar_format = self.build_pbar_format()
        pbar_format = pbar_format.replace('{bar}', '{bar'+':'+str(self.size)+'}')
        
        return tqdm.tqdm(
            desc=desc,
            bar_format=pbar_format,
            unit=' steps',
            ascii=self.ascii,
            total=total,
            leave=leave,
            position=position)
        
    def update_pbar_metrics(self, pbar, metrics):
        """ Updates the progress bar with the given metrics. """
        pbar.set_postfix(ordered_dict=metrics)
        
    def build_pbar_format(self) -> str:
        """ 
        Returns the progress bar format
        
        Inherit from this class and override this function to define another
        progress bar format (e.g Keras format)
        """
        l_bar = self.l_bar
        if l_bar is None:
            l_bar = ''
            if self.show_desc:
                l_bar += '{desc}:'
            if self.show_percentage:
                l_bar += f' {self.percentage_str}%'
            l_bar += ' ' + self.pbar_frames[0]
           
        r_bar = self.r_bar 
        if r_bar is None:
            r_bar = self.pbar_frames[1]
            if self.show_n_fmt:
                r_bar += f' {self.n_fmt_str}'
                if self.show_total_fmt:
                    r_bar += f'/{self.total_fmt_str}'
            
            if self.show_elapsed_time or self.show_rate:
                r_bar += ' ['
                if self.show_elapsed_time:
                    r_bar += self.elapsed_time_str 
                    if self.show_remaining_time:
                        r_bar += f'<{self.remaining_time_str}'
                
                if self.show_rate:
                    if self.show_elapsed_time:
                        r_bar += ', '
                    r_bar += self.rate_str
                r_bar += self.postfix_str + "]"
            else:
                r_bar += self.postfix_str
                    
        pbar_format = l_bar + '{bar}' + r_bar
        return pbar_format
        
    def is_last_epoch(self, trainer) -> bool:
        """ Checks if the last epoch is reached. """
        return trainer.current_epoch + 1 == trainer.max_epochs
    
    # ============== Fit ==============
    def on_train_epoch_start(self, trainer, model):
        if self.pbar is not None: self.pbar.close()
        desc = f"Epoch {trainer.current_epoch + 1}/{trainer.max_epochs}"
        if self.desc_above:
            print(desc); desc = ''
        self.pbar = self.create_pbar(desc=desc, total=trainer.num_training_batches, leave=True)
    
    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx) -> None:
        self.pbar.update(1)
        self.update_pbar_metrics(self.pbar, trainer.get_step_metrics())
        
    def on_train_epoch_end(self, trainer, model) -> None:
        self.update_pbar_metrics(self.pbar, trainer.get_step_metrics())

    def on_validation_batch_end(self, trainer, model, outputs, batch, batch_idx) -> None:
        self.update_pbar_metrics(self.pbar, trainer.get_step_metrics())

    def on_validation_epoch_end(self, trainer, model) -> None:
        self.update_pbar_metrics(self.pbar, trainer.get_step_metrics())
        
    def on_fit_epoch_end(self, trainer, model):
        self.pbar.close()
        
    # ============== Predict ==============
    def on_predict_epoch_start(self, trainer, model) -> None:
        desc = f"Predicting"
        if self.desc_above:
            print(desc); desc = ''
        self.predict_pbar = self.create_pbar(desc=desc, total=trainer.num_predict_batches, leave=True)
        
    def on_predict_batch_end(self, trainer, model, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        self.predict_pbar.update(1)
        self.update_pbar_metrics(self.predict_pbar, trainer.get_step_metrics())
        
    def on_predict_epoch_end(self, trainer, model) -> None:
        self.update_pbar_metrics(self.predict_pbar, trainer.get_step_metrics())
        self.predict_pbar.close()


class SimpleProgressBar(TQDMProgressBar):
    
    def __init__(self, pbar_size:int=30, ascii='.>=', desc_above=False,
                 show_desc=True, show_elapsed_time=True, show_remaining_time=False, show_rate=False,
                 show_postfix=True, show_n_fmt=True, show_total_fmt=True, show_percentage=True,
                 pbar_frames=('[', ']'), l_bar=None, r_bar=None) -> None:
        super().__init__(pbar_size, ascii, desc_above, show_desc, show_elapsed_time, show_remaining_time,
                         show_rate, show_postfix, show_n_fmt, show_total_fmt, show_percentage, pbar_frames,
                         l_bar, r_bar)

        
class KerasProgressBar(TQDMProgressBar):
    pass