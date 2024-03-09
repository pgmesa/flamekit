
import tqdm
from flamekit.callbacks import Callback


class ProgressBar(Callback):
    """ Base class for Progress Bars """
    
    def __init__(self, desc_above=False, show_desc=True) -> None:
        super().__init__()
        self.desc_above = desc_above
        self.show_desc = show_desc
    
    def create_pbar(self, desc, total):
        """ Creates the progress bar. """
        raise NotImplementedError
        
    def update_pbar_metrics(self, pbar, metrics):
        """ Updates the progress bar with the given metrics. """
        raise NotImplementedError
    
    def is_last_epoch(self, trainer) -> bool:
        """ Checks if the last epoch is reached. """
        return trainer.current_epoch + 1 == trainer.max_epochs
    
    # ============== Fit ==============
    def on_train_epoch_start(self, trainer, model):
        if self.pbar is not None: self.pbar.close()
        desc = None
        if self.show_desc:
            desc = f"Epoch {trainer.current_epoch + 1}/{trainer.max_epochs}"
            if self.desc_above:
                print(desc)
                desc = None
        self.pbar = self.create_pbar(desc=desc, total=trainer.num_training_batches)
    
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
        desc = None
        if self.show_desc:
            desc = f"Predicting"
            if self.desc_above:
                print(desc)
                desc = None 
        self.predict_pbar = self.create_pbar(desc=desc, total=trainer.num_predict_batches)
        
    def on_predict_batch_end(self, trainer, model, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        self.predict_pbar.update(1)
        self.update_pbar_metrics(self.predict_pbar, trainer.get_step_metrics())
        
    def on_predict_epoch_end(self, trainer, model) -> None:
        self.update_pbar_metrics(self.predict_pbar, trainer.get_step_metrics())
        self.predict_pbar.close()
        

class TQDMProgressBar(ProgressBar):
    """ TQDM based Progress bar """
    
    def __init__(self, pbar_size:int=30, ascii=None, desc_above=False,
                 show_desc=True, show_elapsed_time=True, show_remaining_time=True, show_rate=True,
                 show_postfix=True, show_n_fmt=True, show_total_fmt=True, show_percentage=True,
                 pbar_frames=('|','|'), l_bar=None, r_bar=None) -> None:
        """ 
        Customizable terminal progress bar using tqdm.
        
        default tqdm l_bar = '{desc}: {percentage:3.0f}% |'
        default tqdm r_bar = '| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
        
        Args:
            pbar_size (int, optional): The size of the progress bar. Defaults to 30.
            ascii (str, optional): The ascii characters to use for the progress bar. Defaults to None.
            desc_above (bool, optional): If True, the description is displayed above the progress bar. Defaults to False.
            show_desc (bool, optional): If True, the description is displayed. Defaults to True.
            show_elapsed_time (bool, optional): If True, the elapsed time is displayed. Defaults to True.
            show_remaining_time (bool, optional): If True, the remaining time is displayed. Defaults to True.
            show_rate (bool, optional): If True, the rate is displayed. Defaults to True.
            show_postfix (bool, optional): If True, the postfix is displayed. Defaults to True.
            show_n_fmt (bool, optional): If True, the number format is displayed. Defaults to True.
            show_total_fmt (bool, optional): If True, the total format is displayed. Defaults to True.
        
        Returns:
            None.
        
        Some values for ascii parameter:
        >>> - ' >='
            - '.>='
            - ' ▖▘▝▗▚▞█'
            - '░▒█'
            - ' ▁▂▃▄▅▆▇█'
        """ 
        super().__init__(desc_above=desc_above, show_desc=show_desc)
        self.ascii = ascii
        self.l_bar = l_bar
        self.r_bar = r_bar
        self.size = pbar_size
        self.pbar_frames = pbar_frames
        
        self.desc_str = '{desc}'
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
    
    def create_pbar(self, desc, total) -> tqdm.tqdm:
        """ Creates the progress bar. """
        pbar_format = self.build_pbar_format()
        pbar_format = pbar_format.replace('{bar}', '{bar'+':'+str(self.size)+'}')
        return tqdm.tqdm(
            desc=desc,
            bar_format=pbar_format,
            unit=' steps',
            ascii=self.ascii,
            total=total)
        
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
            if self.show_desc and not self.desc_above:
                l_bar += self.desc_str + ':'
            if self.show_percentage:
                if self.show_desc and not self.desc_above:
                    l_bar += ' '
                l_bar += f'{self.percentage_str}%'
            
            if len(l_bar) > 0: l_bar += ' '
            l_bar += self.pbar_frames[0]
           
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
        
    
class KerasProgressBar(TQDMProgressBar):
    """ 
    Tries to replicate Keras Progress bar design using TQDM.
    """
    
    def __init__(self, pbar_size:int=30, ascii='.>=', desc_above=True, show_desc=True, show_elapsed_time=True,
                 show_rate=True, show_postfix=True, show_n_fmt=True, show_total_fmt=True,
                 pbar_frames=('[', ']')) -> None:
        super().__init__(pbar_size=pbar_size, ascii=ascii, desc_above=desc_above, show_desc=show_desc,
                         show_elapsed_time=show_elapsed_time, show_rate=show_rate, show_postfix=show_postfix,
                         show_n_fmt=show_n_fmt, show_total_fmt=show_total_fmt, show_percentage=False, 
                         pbar_frames=pbar_frames, l_bar=None, r_bar=None)
    
    def build_pbar_format(self) -> str:
        l_bar = self.l_bar
        if l_bar is None:
            l_bar = ''
            if self.show_desc and not self.desc_above:
                l_bar += self.desc_str + ':'
                
            if self.show_n_fmt:
                if self.show_desc and not self.desc_above:
                    l_bar += ' '
                l_bar += f'{self.n_fmt_str}'
                if self.show_total_fmt:
                    l_bar += f'/{self.total_fmt_str}'
                
            if len(l_bar) > 0: l_bar += ' '
            l_bar += self.pbar_frames[0]
           
        r_bar = self.r_bar 
        if r_bar is None:
            r_bar = self.pbar_frames[1] + ' -'
            
            if self.show_elapsed_time:
                r_bar += ' ' + self.elapsed_time_str
            
            if self.show_rate:
                r_bar += ' ' + self.rate_str
            
            if self.show_postfix:    
                r_bar += self.postfix_str
            
        pbar_format = l_bar + '{bar}' + r_bar
        return pbar_format