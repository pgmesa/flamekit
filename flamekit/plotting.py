
import numpy as np
import matplotlib.pyplot as plt


SIZE_SMALL = 8
SIZE_DEFAULT = 10
SIZE_LARGE = 14

style_rc_params = {    
    'font.weight': "normal",
    'font.family': "DejaVu Sans",
    'font.size': SIZE_DEFAULT,
    'axes.titlesize': SIZE_DEFAULT,
    'axes.labelsize': SIZE_DEFAULT,
    'xtick.labelsize': SIZE_DEFAULT,
    'ytick.labelsize': SIZE_DEFAULT,
    "axes.grid": True,
    "legend.fontsize": SIZE_DEFAULT,
    #"axes.grid.axis": "y",
}

def plot_curves(ax:plt.Axes, x, ys:dict, colors:list=None, add_label_text=False) -> plt.Axes:
    """ 
    Plot multiple curves in a single axis.

    Args:
        ax (plt.Axes): Matplotlib axis object.
        x (list or np.ndarray): X-axis values.
        ys (dict): Dictionary with curve labels as keys and curve values as values.
        colors (list, optional): List of colors to use for each curve. If not provided,
            default matplotlib colors will be used.
        add_label_text (bool): Whether to add label text to the plot instead of using the
            legend. Defaults to False.

    Returns:
        plt.Axes: The axis object with the plotted curves.
    """
    for i, (label, y) in enumerate(ys.items()):
        # Line
        color = colors[i] if colors else None
        # If len(y) < len(x), padd sequence with nan values
        if len(y) < len(x):
            y_pad = np.full((len(x),), np.nan)
            y_pad[:len(y)] = y
            y = y_pad
        ax.plot(x, y, label=label, marker=".", linewidth=2, markersize=8, color=color)

        # Text
        if add_label_text:
            ax.text(
                x[-1] * 1.01, y[-1], label,
                color=ax.lines[i].get_color(), 
                fontweight="bold",
                horizontalalignment="left",
                verticalalignment="center",
            )

    # Change color of right and top spines (axis lines)
    if not add_label_text:
        ax.spines["right"].set_color("grey")
        ax.spines["top"].set_color("grey")
    else:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    
    # Add legend
    if not add_label_text: 
        ax.legend()
    return ax


def plot_curve_groups(x, groups:list[dict], ncols=3, colors=None, 
                      add_label_text=False, dest_path=None, callback:callable=None):
    """ 
    Plot multiple groups of curves in subplots.
    
    Args:
        x (list or np.ndarray): X-axis values.
        groups (list of dict): List of dictionaries, where each dictionary contains
            curve labels as keys and curve values as values.
        ncols (int): Number of columns in the subplot grid.
        colors (list, optional): List of colors to use for plotting. If not provided, default 
                matplotlib colors will be used. Can be a list of lists, containing the colors for 
                each curve group or a single list with same colors to use for all groups.
        add_label_text (bool): Whether to add label text to the plot instead of using the legend.
            Defaults to False.
        dest_path (str, optional): Path to save the figure.
        callback (callable, optional): Function to customize the axis for each group of curves.

    Returns:
        tuple: A tuple containing the figure and a list of axes.
    """
    # Estimate best number of rows and columns
    nrows = int(np.ceil(len(groups) / ncols))
    # Plot each group of curves in a different subplot
    with plt.rc_context(style_rc_params):
        fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(8*ncols, 5*nrows))
        axes = axes.flatten(); diff = len(axes) - len(groups)
        groups = groups + [None]*diff
        for i, (group, ax) in enumerate(zip(groups, axes)):
            ax = axes[i]
            if group is not None:
                # Define colors
                c = None
                if colors is not None:
                    c = colors[0]
                    if isinstance(c, str):
                        c = colors
                    elif isinstance(c, list) or isinstance(c, tuple):
                        assert len(colors) == len(groups) - diff, "Number of colors must match number of groups"
                        c = colors[i]
                    else:
                        raise ValueError(f"Invalid type for colors: {type(c)}")
                # Callback function to customize axis
                if callback:
                    callback(i, ax, x, group, c)
                # Plot curves
                ax = plot_curves(ax, x, group, colors=c, add_label_text=add_label_text)
            else:
                # Remove unused axes
                fig.delaxes(ax)
        # Save figure
        if dest_path is not None:
            # Save PNG with plotted curves
            plt.savefig(dest_path, bbox_inches='tight', dpi=300)
        return fig, axes


def plot_results(results:list[dict], names:list[str]=None, colors=None, filter_keys:callable=None,
                        add_label_text=False, dest_path=None, show=True, verbose=True):
    """  
    Plot the results of multiple experiments in the same plot.

    Args:
        results (list of dict): List of dictionaries containing the results to plot.
        names (list of str, optional): List of names for each result. If not provided,
            default names will be generated.
        colors (list, optional): List of colors to use for each curve. If not provided,
            default matplotlib colors will be used. Check 'plot_curve_groups' for more information.
        filter_keys (callable, optional): Function to filter the keys to display in the plot.
        add_label_text (bool): Whether to add label text to the plot instead of using the legend.
            Defaults to False.
        dest_path (str, optional): Path to save the figure.
        show (bool): Whether to show the plot.
        verbose (bool): Whether to print additional information.

    Returns:
        None
    """
    if names is None:
        names = ["Result " + str(i+1) for i in range(len(results))]
    assert len(results) == len(names), "Number of results and names must match"
    column_names = [set(r.keys()) for r in results]
    # Find columns common to all dictionaries
    common_keys = list(set.intersection(*column_names))
    common_keys.sort()
    keys_to_display = common_keys
    if filter_keys is not None:
        keys_to_display = filter_keys(keys_to_display)
        
    if verbose:
        print(f"Keys common to all results: {common_keys}")
        print(f"Keys to display: {keys_to_display}")
        
    groups = []
    for metric in keys_to_display:
        group = {names[i]:dic[metric] for i, dic in enumerate(results)}
        groups.append(group)
    
    # Plot results
    def customize_axis(idx:int, ax, x, group:dict, colors):
        title = keys_to_display[idx]
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.set_xticks(x)
    
    nelements = [len(r[common_keys[0]]) for r in results]
    x = range(1, max(nelements)+1)
            
    fig, axes = plot_curve_groups(
        x, groups, callback=customize_axis,
        colors=colors, add_label_text=add_label_text, dest_path=dest_path)
    if show:
        plt.tight_layout()
        plt.show()
    else:
        plt.close(fig)

    
    